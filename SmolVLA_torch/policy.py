import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SmolVLMWithExpertModel, create_sinusoidal_pos_embedding

sys.path.append(os.path.dirname(__file__))
from preprocess import pad_tensor, pad_vector, resize_with_pad, normalize_to_minus1_1

class SmolVLA(nn.Module):
    def __init__(
        self,
        action_dim=14,
        state_dim=14,
        chunk_size=50,
        vlm_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        device="cuda",
        prefix_length=-1,
        max_state_dim=None,
        max_action_dim=None,
        resize_imgs_with_padding=None,
        add_image_special_tokens=False,
        use_cache=True,
        rtc_processor=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        self.prefix_length = prefix_length
        self.max_state_dim = max_state_dim or state_dim
        self.max_action_dim = max_action_dim or action_dim
        self.resize_imgs_with_padding = resize_imgs_with_padding
        self.add_image_special_tokens = add_image_special_tokens
        self.use_cache = use_cache
        self.rtc_processor = rtc_processor
        
        # Initialize the core architecture
        self.model = SmolVLMWithExpertModel(vlm_model_id=vlm_model_id)
        self.model.to(device)
        
        # Projection layers
        expert_dim = self.model.expert_hidden_size
        vlm_dim = self.model.config.text_config.hidden_size
        
        # State -> VLM Dimension
        self.state_proj = nn.Linear(self.max_state_dim, vlm_dim)
        
        # Action In (Action + Time -> Hidden Dimension)
        self.action_in_proj = nn.Linear(self.max_action_dim, expert_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(expert_dim * 2, expert_dim),
            nn.SiLU(),
            nn.Linear(expert_dim, expert_dim)
        )
        
        # Action Out (Hidden Dimension -> Velocity Dimension)
        self.action_out_proj = nn.Linear(expert_dim, self.max_action_dim)
        
        # Freeze VLM (Standard Protocol)
        self._freeze_vlm()

    def _freeze_vlm(self):
        """Freeze VLM parameters, only train Expert and projection layers"""
        for param in self.model.vlm.parameters():
            param.requires_grad = False
        # Vision tower is part of VLM, so it's frozen too.
        print("VLM backbone frozen.")

    def preprocess_images(self, images):
        """
        Minimal preprocessing: move to correct device.
        Real resizing/normalization is handled by HF AutoProcessor inside the VLM.
        """
        if images is None:
            raise ValueError("images is None")
        return images.to(self.device)

    def embed_prefix(self, images, text_tokens, state):
        """
        Build Prefix Embedding: [Image_Tokens, Text_Tokens, State_Token]
        This is the input to the VLM.
        Returns embeddings, padding mask, and attention mask (1 means
        this token should not attend to later blocks).
        """
        # Image Embeds
        img_embs = self.model.embed_image(images)  # [B, N_Img, D]
        img_embs = img_embs * (img_embs.shape[-1] ** 0.5)

        # Text Embeds
        txt_embs = self.model.embed_text(text_tokens)
        txt_embs = txt_embs * (txt_embs.shape[-1] ** 0.5)

        # State Embeds
        state_embs = self.state_proj(state).unsqueeze(1)  # [B, 1, D]

        # Concatenate
        prefix_embs = torch.cat([img_embs, txt_embs, state_embs], dim=1)

        # Masks
        b, _, _ = prefix_embs.shape
        pad_mask = torch.ones((b, prefix_embs.shape[1]), device=self.device, dtype=torch.bool)

        att_mask = torch.cat(
            [
                torch.zeros(img_embs.shape[:2], device=self.device, dtype=torch.bool),
                torch.zeros(txt_embs.shape[:2], device=self.device, dtype=torch.bool),
                torch.ones(state_embs.shape[:2], device=self.device, dtype=torch.bool),
            ],
            dim=1,
        )

        # Optional padding to fixed prefix length for cache-friendly shapes
        if self.prefix_length > 0 and prefix_embs.shape[1] < self.prefix_length:
            target_len = self.prefix_length
            prefix_embs = pad_tensor(prefix_embs, target_len, pad_value=0)
            pad_mask = pad_tensor(pad_mask, target_len, pad_value=0)
            att_mask = pad_tensor(att_mask, target_len, pad_value=0)

        return prefix_embs, pad_mask, att_mask

    def embed_suffix(self, noisy_actions, time):
        """
        Build Suffix Embedding: [Action_Time_Tokens]
        This is the input to the Expert.
        Returns embeddings, padding mask, and attention mask (1 => causal block).
        """
        # Action Proj
        act_emb = self.action_in_proj(noisy_actions)

        # Time Embed
        time_emb = create_sinusoidal_pos_embedding(
            time, self.model.expert_hidden_size, device=self.device
        ).to(act_emb.dtype)

        # Broadcast Time: [B, D] -> [B, 1, D] -> [B, Seq, D]
        time_emb = time_emb.unsqueeze(1).expand_as(act_emb)

        # Fuse Action + Time
        fusion = torch.cat([act_emb, time_emb], dim=-1)
        suffix_embs = self.time_mlp(fusion)

        b, l, _ = suffix_embs.shape
        pad_mask = torch.ones((b, l), device=self.device, dtype=torch.bool)

        # suffix tokens use causal mask downstream -> mark as 1
        att_mask = torch.ones((b, l), device=self.device, dtype=torch.bool)

        return suffix_embs, pad_mask, att_mask

    def make_att_2d_masks(self, pad_masks, att_masks):
        """
        Build 2D attention masks following official SmolVLA logic.
        Tokens can attend to valid tokens whose cumulative att_mask is
        <= their own. att_mask acts as block identifiers for causal groups.
        """
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks

    def forward(self, images, text_tokens, state, actions):
        """
        Training Step: Flow Matching Loss
        """
        B = actions.shape[0]

        # Preprocess images and pad state/actions to max dims
        images = self.preprocess_images(images)
        state = pad_vector(state, self.max_state_dim)
        actions = pad_vector(actions, self.max_action_dim)
        
        # Sample Time & Noise (clipped away from 0/1 for stability)
        time = torch.distributions.Beta(1.5, 1.0).sample((B,)).to(self.device)
        time = time * 0.999 + 0.001
        noise = torch.randn_like(actions)
        
        # Flow Matching Interpolation (Rectified Flow)
        # x_t = t * x_1 + (1-t) * x_0
        # x_0 = data (actions), x_1 = noise
        # Note: Paper uses u_t = x_1 - x_0 target
        t_expand = time.view(B, 1, 1)
        noisy_actions = t_expand * noise + (1 - t_expand) * actions
        target_velocity = noise - actions
        
        # Embeddings
        prefix_embs, prefix_pad_mask, prefix_att_mask = self.embed_prefix(images, text_tokens, state)
        suffix_embs, suffix_pad_mask, suffix_att_mask = self.embed_suffix(noisy_actions, time)
        
        # Attention Mask (official style)
        pad_mask = torch.cat([prefix_pad_mask, suffix_pad_mask], dim=1)
        att_mask = torch.cat([prefix_att_mask, suffix_att_mask], dim=1)
        att_mask_2d = self.make_att_2d_masks(pad_mask, att_mask)
        
        # Position IDs (Simple cumsum over pad mask)
        total_mask = pad_mask
        position_ids = torch.cumsum(total_mask, dim=1) - 1
        
        # Model Forward
        inputs = [prefix_embs, suffix_embs]
        _, expert_out = self.model(
            input_embeds=inputs,
            attention_mask=att_mask_2d,
            position_ids=position_ids
        )
        
        # Prediction & Loss
        # expert_out corresponds to suffix part
        pred_velocity = self.action_out_proj(expert_out)
        # crop to original action_dim
        pred_velocity = pred_velocity[:, :, : self.action_dim]
        target_velocity = target_velocity[:, :, : self.action_dim]

        loss = F.mse_loss(pred_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def inference(self, images, text_tokens, state, num_steps=10):
        """
        Inference Step: Euler Integration
        """
        B = images.shape[0]
        
        # Prepare Prefix (Fixed)
        images = self.preprocess_images(images)
        state = pad_vector(state, self.max_state_dim)
        prefix_embs, prefix_pad_mask, prefix_att_mask = self.embed_prefix(images, text_tokens, state)

        # Prefix-only masks & positions for caching
        prefix_att_2d = self.make_att_2d_masks(prefix_pad_mask, prefix_att_mask)
        prefix_pos_ids = torch.cumsum(prefix_pad_mask, dim=1) - 1
        prefix_cache = self.model.build_prefix_cache(prefix_embs, prefix_att_2d, prefix_pos_ids)
        
        # Start from Noise (x_1)
        x_t = torch.randn((B, self.chunk_size, self.max_action_dim), device=self.device)
        
        dt = 1.0 / num_steps
        past_kv = None
        
        # Euler Loop (t: 1.0 -> 0.0)
        for i in range(num_steps):
            t_val = 1.0 - i * dt
            time = torch.full((B,), t_val, device=self.device)
            
            # Embed Current Actions
            suffix_embs, suffix_pad_mask, suffix_att_mask = self.embed_suffix(x_t, time)
            
            # Masks & Position IDs
            pad_mask = torch.cat([prefix_pad_mask, suffix_pad_mask], dim=1)
            att_mask = torch.cat([prefix_att_mask, suffix_att_mask], dim=1)
            att_mask_2d = self.make_att_2d_masks(pad_mask, att_mask)
            total_mask = pad_mask
            position_ids = torch.cumsum(total_mask, dim=1) - 1
            
            # Model Forward with cached prefix
            inputs = {
                "prefix_cache": prefix_cache,
                "suffix_embs": suffix_embs,
                "past_key_values": past_kv,
                "use_cache": self.use_cache,
                "fill_kv_cache": self.use_cache,
            }
            _, expert_out, kv_out = None, None, None
            outputs = self.model(
                input_embeds=inputs,
                attention_mask=att_mask_2d,
                position_ids=position_ids
            )
            if isinstance(outputs, tuple) and len(outputs) == 2:
                (_, expert_out), past_kv = outputs
            else:
                (_, expert_out), past_kv = outputs, None
            
            pred_velocity = self.action_out_proj(expert_out)

            # RTC hook (optional)
            if self.rtc_processor is not None:
                pred_velocity = self.rtc_processor(
                    x_t=x_t,
                    v_t=pred_velocity,
                    time=t_val,
                    step=i,
                    num_steps=num_steps,
                )
            
            # Euler Update: x_{t-dt} = x_t - v * dt
            x_t = x_t - pred_velocity * dt
            
        # return cropped to original action_dim
        return x_t[:, :, : self.action_dim]


class SimpleRTCProcessor:
    """
    Minimal RTC hook to allow injection of latency/leftover handling without lerobot.
    Users can subclass/override __call__ for custom behavior.
    """

    def __init__(self, inference_delay=None, prev_chunk_left_over=None, execution_horizon=None):
        self.inference_delay = inference_delay
        self.prev_chunk_left_over = prev_chunk_left_over
        self.execution_horizon = execution_horizon
        self.buffer = prev_chunk_left_over

    def __call__(self, x_t, v_t, time, step, num_steps):
        # Basic behavior: if buffer exists and inference_delay > 0, use buffer for initial steps
        if self.inference_delay and step < self.inference_delay and self.buffer is not None:
            # consume buffer (assumed shape compatible with v_t)
            take = min(self.buffer.shape[1], v_t.shape[1])
            v_buf = self.buffer[:, :take]
            self.buffer = self.buffer[:, take:] if take < self.buffer.shape[1] else None
            # pad if buffer shorter
            if v_buf.shape[1] < v_t.shape[1]:
                pad_len = v_t.shape[1] - v_buf.shape[1]
                pad = torch.zeros_like(v_t[:, :pad_len])
                v_buf = torch.cat([v_buf, pad], dim=1)
            return v_buf
        return v_t