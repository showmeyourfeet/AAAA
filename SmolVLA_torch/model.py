import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
from transformers import AutoModelForImageTextToText, AutoConfig, AutoModel, AutoProcessor

# -----------------------------------------------------------------------------
# Assistant functions (Positional Embedding & RoPE)
# -----------------------------------------------------------------------------

def apply_rope(x, positions, max_wavelength=10_000):
    """Apply Rotary Positional Embedding (RoPE)"""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)

def create_sinusoidal_pos_embedding(time, dimension, min_period=4e-3, max_period=4.0, device="cpu"):
    """Create sinusoidal positional embedding for time steps"""
    half_dim = dimension // 2
    fraction = torch.linspace(0.0, 1.0, half_dim, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


# -----------------------------------------------------------------------------
# SmolVLM + Expert Mixture Model
# -----------------------------------------------------------------------------

class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", # default vlm backbone,
        num_vlm_layers = 16, # layers skipping: only the first num_vlm_layers will be used
        num_experts_layers = 12, # experts layers
        expert_width_multiplier = 0.75, # expert width multiplier relative to vlm width
        self_attn_every_n_layers = 2, # Interleaving frequency: self attention every n layers, others are cross attention
    ):
        super().__init__()

        # load vlm model
        print(f"Loading VLM backbone: {vlm_model_id} ...")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            vlm_model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.config = self.vlm.config
        # Processor for vision/text normalization & resizing
        self.processor = AutoProcessor.from_pretrained(vlm_model_id)

        # Layers skipping
        # truncate the layers list.
        self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.vlm.model.text_model.layers)
        print(f"VLM truncated to {self.num_vlm_layers} layers.")

        # Construct Action Expert
        expert_config = copy.deepcopy(self.config.text_config)
        hidden_size = expert_config.hidden_size
        expert_config.hidden_size = int(expert_width_multiplier * hidden_size)
        expert_config.intermediate_size = get_intermediate_size(expert_config.hidden_size)
        expert_config.num_hidden_layers = num_experts_layers
        self.expert_hidden_size = expert_config.hidden_size

        # Init Expert model (Transfromer Decoder-only)
        self.lm_expert = AutoModel.from_config(expert_config)
        # remove the useless Embeddings
        self.lm_expert.embed_tokens = None

        self.num_experts_layers = num_experts_layers
        self.self_attn_every_n_layers = self_attn_every_n_layers

        # Adjust the cross attention layers in Expert model
        # Cause within the cross attention, the key/value is from the VLM backbone (higher dimension), while the query is from the Expert model (lower dimension).
        # The dimension should be adjusted.
        # Note: for Llama-based AutoModel, transformer blocks live in `self.lm_expert.layers` (no `.model` attribute).
        for layer_idx, layer in enumerate(self.lm_expert.layers):
            # if the current layer is the cross attention layer rather than the self attention layer,
            if self_attn_every_n_layers > 0 and layer_idx % self_attn_every_n_layers != 0:
                # reset the linear layers to match the dim(head_dim*num_heads) in the VLM backbone
                vlm_dim = self.config.text_config.num_key_value_heads * self.config.text_config.head_dim
                expert_kv_dim = expert_config.num_key_value_heads * expert_config.head_dim

                layer.self_attn.k_proj = nn.Linear(vlm_dim, expert_kv_dim, bias=expert_config.attention_bias)
                layer.self_attn.v_proj = nn.Linear(vlm_dim, expert_kv_dim, bias=expert_config.attention_bias)

        # Dedicated cross-attn projection lists to guarantee dimensions at runtime
        self.vlm_dim = self.config.text_config.hidden_size
        self.vlm_kv_dim = self.config.text_config.num_key_value_heads * self.config.text_config.head_dim
        self.expert_kv_dim = expert_config.num_key_value_heads * expert_config.head_dim
        self.cross_k_projs = nn.ModuleList()
        self.cross_v_projs = nn.ModuleList()
        for layer_idx in range(len(self.lm_expert.layers)):
            if self_attn_every_n_layers > 0 and layer_idx % self_attn_every_n_layers != 0:
                # Cross-attn: input dim = VLM hidden size, output dim = expert kv dim
                self.cross_k_projs.append(
                    nn.Linear(self.vlm_dim, self.expert_kv_dim, bias=expert_config.attention_bias)
                )
                self.cross_v_projs.append(
                    nn.Linear(self.vlm_dim, self.expert_kv_dim, bias=expert_config.attention_bias)
                )
            else:
                self.cross_k_projs.append(None)
                self.cross_v_projs.append(None)

    def embed_image(self, image):
        """
        Extract the image features using the VLM Tower.
        We use the official AutoProcessor to ensure correct resizing and normalization.
        For the video SmolVLM, AutoProcessor returns pixel_values of shape [B, T, C, H, W].
        We treat each sample as a "video" of T frames, and flatten (B, T) for the vision
        tower, then reshape back to [B, T * N_img, D] for downstream use.
        """
        # image: [B, C, H, W] torch tensor in [0,1]
        if not isinstance(image, torch.Tensor):
            raise ValueError("embed_image expects a torch.Tensor of shape [B, C, H, W]")

        # Move to CPU and convert to numpy [B, H, W, C]
        imgs = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        # For video processor, pass list of videos; each video here has 1 frame.
        videos = [[img] for img in imgs]  # List of length B, each is [H,W,C]

        processed = self.processor(
            videos=videos,
            return_tensors="pt",
            do_rescale=False,  # inputs already in [0,1]
        )
        pixel_values = processed["pixel_values"]

        # For SmolVLM2-Video-Instruct the processor returns 5D [B, T, C, H, W].
        # Flatten (B, T) for the vision tower, then reshape embeddings back.
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected 5D pixel_values [B, T, C, H, W], got shape {pixel_values.shape}")

        b, t, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * t, c, h, w)

        vision_dtype = getattr(self.vlm.model.vision_model, "dtype", torch.bfloat16)
        pixel_values = pixel_values.to(device=self.vlm.device, dtype=vision_dtype)

        vision_outputs = self.vlm.model.vision_model(pixel_values=pixel_values)
        image_hidden = vision_outputs.last_hidden_state  # [B*T, N_img, D]
        image_embeds = self.vlm.model.connector(image_hidden)  # [B*T, N_img, D']

        # Reshape back to [B, T * N_img, D']
        b_t, n_img, dim = image_embeds.shape
        if b_t != b * t:
            raise ValueError(f"Unexpected vision output shape {image_embeds.shape}, expected first dim {b*t}")
        image_embeds = image_embeds.view(b, t * n_img, dim)
        return image_embeds

    def embed_text(self, input_ids):
        return self.vlm.model.text_model.get_input_embeddings()(input_ids)

    def get_attention_interface(self):
        # simple sdpa interface
        def eager_attention_forward(
            attention_mask, 
            batch_size, 
            head_dim, 
            query_states, 
            key_states, 
            value_states
        ):
            # Q, K, V are already [B, Num_Heads, Seq_Len, Head_Dim] via reshaping outside
            # But the inputs to this func are flattened. Let's rely on standard PyTorch SDPA 
            # Note: The official code does manual reshaping. Here we simplify slightly for clarity.
            
            # Reshape inputs to [B, Heads, Seq, Dim]
            # ... (Implementation details omitted for brevity, logic follows official code)
            # For robustness, we use the manual logic from official implementation:
            
            num_heads = query_states.shape[1] # Actually concatenated
            # Re-implementation of manual attention logic to match tensor shapes
            # [B, Seq, H*D] -> [B, Seq, H, D] -> [B, H, Seq, D]
            
            # Simplified Logic:
            # Assume query_states is [B, Seq, H_q*D]
            # key_states is [B, Seq, H_kv*D]
            return F.scaled_dot_product_attention(
                query_states, key_states, value_states, 
                attn_mask=attention_mask, dropout_p=0.0
            )
        return self.eager_attention_forward # Use the complex one defined below

    def eager_attention_forward(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        """
        Attention that supports different head counts (GQA) and boolean masks.
        We compute attention in float32 for numerical stability and to avoid
        dtype mismatches, then cast back to the original dtype of query_states.
        """
        orig_dtype = query_states.dtype
        # Upcast to float32 for attention math
        query_states = query_states.to(torch.float32)
        key_states = key_states.to(torch.float32)
        value_states = value_states.to(torch.float32)

        num_q_heads = self.config.text_config.num_attention_heads
        num_kv_heads = self.config.text_config.num_key_value_heads

        if query_states.shape[-1] != (num_q_heads * head_dim):
            curr_hidden = query_states.shape[-1]
            num_q_heads = curr_hidden // head_dim

        b, seq_len_q, _ = query_states.shape
        seq_len_kv = key_states.shape[1]

        q = query_states.view(b, seq_len_q, num_q_heads, head_dim).transpose(1, 2)
        k = key_states.view(b, seq_len_kv, -1, head_dim).transpose(1, 2)
        v = value_states.view(b, seq_len_kv, -1, head_dim).transpose(1, 2)

        if k.shape[1] != q.shape[1]:
            repeat_factor = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if attention_mask.dim() == 2:
            attn_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            attn_mask = attention_mask[:, None, :, :]
        else:
            attn_mask = attention_mask

        att_weights = torch.matmul(q, k.transpose(2, 3)) * (head_dim ** -0.5)

        if attn_mask is not None:
            att_weights = att_weights.masked_fill(~attn_mask.bool(), torch.finfo(att_weights.dtype).min)

        attn_probs = F.softmax(att_weights, dim=-1)
        output = torch.matmul(attn_probs, v)

        output = output.transpose(1, 2).reshape(b, seq_len_q, -1)
        return output.to(orig_dtype)

    def build_prefix_cache(self, prefix_embs, attention_mask, position_ids):
        """
        Run VLM part once and cache normalized hidden states per layer for reuse.
        Returns list of norm_hidden_vlm per layer and final hidden state.
        """
        vlm_layers = self.vlm.model.text_model.layers
        hidden_states_vlm = prefix_embs
        batch_size = hidden_states_vlm.shape[0]
        head_dim = self.config.text_config.head_dim
        norm_cache = []

        for layer_idx, vlm_layer in enumerate(vlm_layers):
            residual_vlm = hidden_states_vlm
            norm_hidden_vlm = vlm_layer.input_layernorm(hidden_states_vlm)
            norm_cache.append(norm_hidden_vlm)

            q_vlm = vlm_layer.self_attn.q_proj(norm_hidden_vlm)
            k_vlm = vlm_layer.self_attn.k_proj(norm_hidden_vlm)
            v_vlm = vlm_layer.self_attn.v_proj(norm_hidden_vlm)

            seq_len_vlm = q_vlm.shape[1]
            pos_ids_vlm = position_ids[:, :seq_len_vlm]

            q_vlm = q_vlm.view(batch_size, seq_len_vlm, -1, head_dim)
            k_vlm = k_vlm.view(batch_size, seq_len_vlm, -1, head_dim)
            v_vlm = v_vlm.view(batch_size, seq_len_vlm, -1, head_dim)

            q_vlm = apply_rope(q_vlm, pos_ids_vlm)
            k_vlm = apply_rope(k_vlm, pos_ids_vlm)

            q_vlm = q_vlm.flatten(2)
            k_vlm = k_vlm.flatten(2)
            v_vlm = v_vlm.flatten(2)

            attn_mask_vlm = attention_mask[:, :seq_len_vlm, :seq_len_vlm]
            vlm_attn_out = self.eager_attention_forward(
                attn_mask_vlm, batch_size, head_dim, q_vlm, k_vlm, v_vlm
            )

            vlm_attn_out = vlm_layer.self_attn.o_proj(vlm_attn_out)
            hidden_states_vlm = residual_vlm + vlm_attn_out

            residual_vlm = hidden_states_vlm
            hidden_states_vlm = vlm_layer.post_attention_layernorm(hidden_states_vlm)
            hidden_states_vlm = vlm_layer.mlp(hidden_states_vlm)
            hidden_states_vlm = residual_vlm + hidden_states_vlm

        hidden_states_vlm = self.vlm.model.text_model.norm(hidden_states_vlm)
        return {"norm_hidden_vlm": norm_cache, "final_hidden_vlm": hidden_states_vlm}

    def forward_cached(
        self,
        prefix_cache,
        suffix_embs,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=False,
        fill_kv_cache=False,
    ):
        """
        Forward pass using cached VLM hidden states; only runs Expert stack.
        If use_cache=True, we can fill or reuse expert KV for incremental decoding.
        """
        expert_layers = self.lm_expert.layers
        norm_cache = prefix_cache["norm_hidden_vlm"]
        batch_size = suffix_embs.shape[0]
        head_dim = self.config.text_config.head_dim
        seq_len_vlm = norm_cache[0].shape[1]

        hidden_states_expert = suffix_embs
        if use_cache and past_key_values is None:
            past_key_values = {}
        kv_cache = past_key_values

        for layer_idx in range(len(norm_cache)):
            expert_layer = expert_layers[layer_idx] if layer_idx < len(expert_layers) else None

            if expert_layer is not None:
                residual_expert = hidden_states_expert
                norm_hidden_expert = expert_layer.input_layernorm(hidden_states_expert)

                is_cross_attn = (self.self_attn_every_n_layers > 0) and (layer_idx % self.self_attn_every_n_layers != 0)

                q_expert = expert_layer.self_attn.q_proj(norm_hidden_expert)
                seq_len_expert = q_expert.shape[1]

                pos_ids_expert = position_ids[:, seq_len_vlm:]
                q_expert = q_expert.view(batch_size, seq_len_expert, -1, head_dim)
                q_expert = apply_rope(q_expert, pos_ids_expert)
                q_expert = q_expert.flatten(2)

                if is_cross_attn:
                    norm_hidden_vlm = norm_cache[layer_idx]
                    k_expert = expert_layer.self_attn.k_proj(norm_hidden_vlm)
                    v_expert = expert_layer.self_attn.v_proj(norm_hidden_vlm)
                    attn_mask_expert = attention_mask[:, seq_len_vlm:, :seq_len_vlm]
                else:
                    k_expert = expert_layer.self_attn.k_proj(norm_hidden_expert)
                    v_expert = expert_layer.self_attn.v_proj(norm_hidden_expert)

                    k_expert = k_expert.view(batch_size, seq_len_expert, -1, head_dim)
                    k_expert = apply_rope(k_expert, pos_ids_expert)
                    k_expert = k_expert.flatten(2)

                    attn_mask_expert = attention_mask[:, seq_len_vlm:, seq_len_vlm:]

                k_expert = k_expert.view(batch_size, -1, self.lm_expert.config.num_key_value_heads, head_dim).flatten(2)
                v_expert = v_expert.view(batch_size, -1, self.lm_expert.config.num_key_value_heads, head_dim).flatten(2)

                # Cache logic: only meaningful for self-attn (causal) when use_cache=True
                if use_cache and not is_cross_attn:
                    if fill_kv_cache:
                        kv_cache[layer_idx] = {"key_states": k_expert, "value_states": v_expert}
                    else:
                        if layer_idx in kv_cache:
                            k_prev = kv_cache[layer_idx]["key_states"]
                            v_prev = kv_cache[layer_idx]["value_states"]
                            k_expert = torch.cat([k_prev, k_expert], dim=1)
                            v_expert = torch.cat([v_prev, v_expert], dim=1)
                        kv_cache[layer_idx] = {"key_states": k_expert, "value_states": v_expert}

                expert_attn_out = self.eager_attention_forward(
                    attn_mask_expert, batch_size, head_dim, q_expert, k_expert, v_expert
                )

                expert_attn_out = expert_layer.self_attn.o_proj(expert_attn_out)
                hidden_states_expert = residual_expert + expert_attn_out

                residual_expert = hidden_states_expert
                hidden_states_expert = expert_layer.post_attention_layernorm(hidden_states_expert)
                hidden_states_expert = expert_layer.mlp(hidden_states_expert)
                hidden_states_expert = residual_expert + hidden_states_expert

        hidden_states_expert = self.lm_expert.norm(hidden_states_expert)
        return [prefix_cache.get("final_hidden_vlm", None), hidden_states_expert], kv_cache

    def forward(
        self,
        input_embeds,
        attention_mask,
        position_ids,
        past_key_values = None,
    ):
        """
        Core Interleaved Forward Pass Loop.
        inputs_embeds: [prefix_embeds, suffix_embeds]
            - prefix: Image + Text + State (Input to VLM)
            - suffix: Action + Timestep (Input to Expert)
        """
        if isinstance(input_embeds, list) and input_embeds[0] is None:
            raise ValueError("prefix embeddings required unless using forward_cached")
        if isinstance(input_embeds, dict):
            # allow prefix cache path
            return self.forward_cached(
                prefix_cache=input_embeds["prefix_cache"],
                suffix_embs=input_embeds["suffix_embs"],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=input_embeds.get("past_key_values"),
                use_cache=input_embeds.get("use_cache", False),
                fill_kv_cache=input_embeds.get("fill_kv_cache", False),
            )

        # Model layers list
        vlm_layers = self.vlm.model.text_model.layers
        expert_layers = self.lm_expert.layers

        # VLM Inputs
        hidden_states_vlm = input_embeds[0]

        # Expert Inputs
        hidden_states_expert = input_embeds[1]

        batch_size = hidden_states_vlm.shape[0]
        head_dim = self.config.text_config.head_dim

        # Loop process
        # Assume the number of Expert layers and truncating VLM layers are same or have a multiple relationship.
        # Here considering they are same (simplest case).
        for layer_idx in range(len(vlm_layers)):
            vlm_layer = vlm_layers[layer_idx]
            expert_layer = expert_layers[layer_idx] if layer_idx < len(expert_layers) else None

            # ------------------- VLM Layer Forward -------------------
            # Keep self attention in VLM layers.
            # pre-Norm
            residual_vlm = hidden_states_vlm
            norm_hidden_vlm = vlm_layer.input_layernorm(hidden_states_vlm)
            # Align dtype with q_proj weights to avoid linear dtype mismatch
            norm_hidden_vlm = norm_hidden_vlm.to(dtype=vlm_layer.self_attn.q_proj.weight.dtype)

            # QKV projections
            q_vlm = vlm_layer.self_attn.q_proj(norm_hidden_vlm)
            k_vlm = vlm_layer.self_attn.k_proj(norm_hidden_vlm)
            v_vlm = vlm_layer.self_attn.v_proj(norm_hidden_vlm)

            # RoPE
            seq_len_vlm = q_vlm.shape[1]
            pos_ids_vlm = position_ids[:, :seq_len_vlm]

            q_vlm = q_vlm.view(batch_size, seq_len_vlm, -1, head_dim)
            k_vlm = k_vlm.view(batch_size, seq_len_vlm, -1, head_dim)
            v_vlm = v_vlm.view(batch_size, seq_len_vlm, -1, head_dim)

            q_vlm = apply_rope(q_vlm, pos_ids_vlm)
            k_vlm = apply_rope(k_vlm, pos_ids_vlm)

            # Flatten back for attention function compatibility
            q_vlm = q_vlm.flatten(2)
            k_vlm = k_vlm.flatten(2)
            v_vlm = v_vlm.flatten(2)

            # VLM Self-Attention
            attn_mask_vlm = attention_mask[:, :seq_len_vlm, :seq_len_vlm]
            vlm_attn_out = self.eager_attention_forward(
                attn_mask_vlm, batch_size, head_dim, q_vlm, k_vlm, v_vlm
            )
            
            # VLM Output Proj & MLP
            vlm_attn_out = vlm_layer.self_attn.o_proj(vlm_attn_out)
            hidden_states_vlm = residual_vlm + vlm_attn_out
            
            residual_vlm = hidden_states_vlm
            hidden_states_vlm = vlm_layer.post_attention_layernorm(hidden_states_vlm)
            # Align dtype with MLP weights to avoid linear dtype mismatch
            hidden_states_vlm = hidden_states_vlm.to(dtype=vlm_layer.mlp.down_proj.weight.dtype)
            hidden_states_vlm = vlm_layer.mlp(hidden_states_vlm)
            hidden_states_vlm = residual_vlm + hidden_states_vlm

            # ------------------- Expert Layer Forward -------------------
            if expert_layer is not None:
                residual_expert = hidden_states_expert
                norm_hidden_expert = expert_layer.input_layernorm(hidden_states_expert)
                # Align dtype with expert q_proj weights
                norm_hidden_expert = norm_hidden_expert.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
                
                # Determine Attention Mode: self-attention or cross-attention
                is_cross_attn = (self.self_attn_every_n_layers > 0) and (layer_idx % self.self_attn_every_n_layers != 0)
                
                # Q (always from Expert)
                q_expert = expert_layer.self_attn.q_proj(norm_hidden_expert)
                seq_len_expert = q_expert.shape[1]
                
                # RoPE for Expert Query
                pos_ids_expert = position_ids[:, seq_len_vlm:]
                q_expert = q_expert.view(batch_size, seq_len_expert, -1, head_dim)
                q_expert = apply_rope(q_expert, pos_ids_expert)
                q_expert = q_expert.flatten(2)

                if is_cross_attn:
                    # Cross Attention: Expert tokens attend to VLM tokens
                    ck = self.cross_k_projs[layer_idx]
                    cv = self.cross_v_projs[layer_idx]
                    if ck is None or cv is None:
                        raise ValueError("Cross projections not initialized for cross-attn layer")
                    # Match dtype with cross-attn projection weights
                    target_dtype = ck.weight.dtype
                    # norm_hidden_vlm: [B, L_pre, vlm_dim]
                    # reshape to 2D for linear, then restore
                    bsz, seq_pre, _ = norm_hidden_vlm.shape
                    kv_in = norm_hidden_vlm.to(dtype=target_dtype).reshape(bsz * seq_pre, self.vlm_dim)
                    k_expert = ck(kv_in).reshape(bsz, seq_pre, self.expert_kv_dim)
                    v_expert = cv(kv_in).reshape(bsz, seq_pre, self.expert_kv_dim)

                    # Cross Attn Mask: Expert tokens attend to VLM tokens
                    attn_mask_expert = attention_mask[:, seq_len_vlm:, :seq_len_vlm]
                else:
                    # Self Attention (Causal): Expert tokens attend to themselves
                    k_expert = expert_layer.self_attn.k_proj(norm_hidden_expert)
                    v_expert = expert_layer.self_attn.v_proj(norm_hidden_expert)

                    # RoPE for self-attention keys
                    k_expert = k_expert.view(batch_size, seq_len_expert, -1, head_dim)
                    k_expert = apply_rope(k_expert, pos_ids_expert)
                    k_expert = k_expert.flatten(2)

                    # self-attention mask (Causal)
                    attn_mask_expert = attention_mask[:, seq_len_vlm:, seq_len_vlm:]
                
                # Reshape K/V for attention
                k_expert = k_expert.view(batch_size, -1, self.lm_expert.config.num_key_value_heads, head_dim).flatten(2)
                v_expert = v_expert.view(batch_size, -1, self.lm_expert.config.num_key_value_heads, head_dim).flatten(2)

                # Expert Attention Execution
                expert_attn_out = self.eager_attention_forward(
                    attn_mask_expert, batch_size, head_dim, q_expert, k_expert, v_expert
                )
                
                # Expert Output Proj & MLP
                expert_attn_out = expert_layer.self_attn.o_proj(expert_attn_out)
                hidden_states_expert = residual_expert + expert_attn_out
                
                residual_expert = hidden_states_expert
                hidden_states_expert = expert_layer.post_attention_layernorm(hidden_states_expert)
                # Align dtype with Expert MLP weights
                hidden_states_expert = hidden_states_expert.to(dtype=expert_layer.mlp.down_proj.weight.dtype)
                hidden_states_expert = expert_layer.mlp(hidden_states_expert)
                hidden_states_expert = residual_expert + hidden_states_expert

        # Final Norms
        hidden_states_vlm = self.vlm.model.text_model.norm(hidden_states_vlm)
        hidden_states_expert = self.lm_expert.norm(hidden_states_expert)

        return [hidden_states_vlm, hidden_states_expert]








            
        



    
