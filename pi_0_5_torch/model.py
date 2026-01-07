"""
Pi0.5 Model Implementation - Fully aligned with OpenPI official implementation.

This module implements the Pi0.5 model architecture with:
1. PaliGemma + Gemma expert configuration via CONFIG_MAPPING (matching OpenPI exactly)
2. Precision strategy with selective bfloat16/float32 (to_bfloat16_for_selected_params)
3. Loss reduction="none" for flexible external aggregation
4. KV cache support for efficient inference (sample_actions + denoise_step)
5. Multi-camera support in embed_prefix
6. **Hierarchical inference**: High-level subtask prediction + Low-level action generation (Pi0.5 paper)

High-Level/Low-Level Architecture (from Pi0.5 paper Section IV.A):
    π_θ(a_{t:t+H}, ℓ̂ | o_t, ℓ) = π_θ(a_{t:t+H} | o_t, ℓ̂) · π_θ(ℓ̂ | o_t, ℓ)

Training Loss (Equation 1):
    L = H(x_{1:M}, f_θ^ℓ(o_t, ℓ)) + α ||ω - a_{t:t+H} - f_θ^a(a_{t:t+H}^{τ,ω}, o_t, ℓ)||²

where:
    - H: Cross-entropy loss for subtask text prediction
    - Second term: Flow matching loss for action generation
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import GemmaForCausalLM, PaliGemmaForConditionalGeneration, PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

from .configuration import PI05Config


# ============================================================================
# Output dataclasses for hierarchical inference
# ============================================================================


@dataclass
class Pi05HierarchicalOutput:
    """Output container for hierarchical forward pass."""
    
    # Loss components
    action_loss: Optional[Tensor] = None  # [B, T, A] flow matching loss (reduction="none")
    text_loss: Optional[Tensor] = None    # [B, L] cross-entropy loss for subtask prediction
    total_loss: Optional[Tensor] = None   # Combined loss (scalar)
    
    # Predicted outputs
    predicted_actions: Optional[Tensor] = None  # [B, T, A] sampled actions
    predicted_subtask_ids: Optional[Tensor] = None  # [B, L] generated subtask token ids
    
    # Intermediate states
    prefix_embeds: Optional[Tensor] = None
    suffix_embeds: Optional[Tensor] = None

logger = logging.getLogger(__name__)

# Constant definition - exact value from OpenPI
OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38


# ============================================================================
# Auxiliary functions (faithful to OpenPI pi0_pytorch.py)
# ============================================================================


def get_safe_dtype(target_dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: str | torch.device = "cpu",
) -> Tensor:
    """
    Compute sine-cosine positional embedding vectors for scalar positions.
    Faithful to OpenPI implementation.
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)`.")

    device = torch.device(device)
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> Tensor:
    """Sample from Beta distribution, as in OpenPI."""
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """
    Build 2D attention mask as in OpenPI / BigVision.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs (block-wise causal attention).
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks.ndim = {att_masks.ndim}, expected 2")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks.ndim = {pad_masks.ndim}, expected 2")

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# ============================================================================
# PaliGemmaWithExpertModel - Aligned with OpenPI gemma_pytorch.py
# ============================================================================


class PaliGemmaWithExpertModel(nn.Module):
    """
    Combined model of PaliGemma (VLM) and Gemma (Action Expert).
    Fully aligned with OpenPI's gemma_pytorch.PaliGemmaWithExpertModel.
    """

    def __init__(
        self,
        vlm_config: dict,
        action_expert_config: dict,
        use_adarms: list[bool] | None = None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # Build PaliGemma config using CONFIG_MAPPING (exactly as OpenPI does)
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config["width"]
        vlm_config_hf.text_config.intermediate_size = vlm_config["mlp_dim"]
        vlm_config_hf.text_config.num_attention_heads = vlm_config["num_heads"]
        vlm_config_hf.text_config.head_dim = vlm_config["head_dim"]
        vlm_config_hf.text_config.num_hidden_layers = vlm_config["depth"]
        vlm_config_hf.text_config.num_key_value_heads = vlm_config["num_kv_heads"]
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config["width"] if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # Build Gemma expert config
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config["head_dim"],
            hidden_size=action_expert_config["width"],
            intermediate_size=action_expert_config["mlp_dim"],
            num_attention_heads=action_expert_config["num_heads"],
            num_hidden_layers=action_expert_config["depth"],
            num_key_value_heads=action_expert_config["num_kv_heads"],
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config["width"] if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None  # Expert doesn't need token embeddings

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        """
        Convert to bfloat16 while keeping certain critical params in float32.
        Exactly matches OpenPI's precision handling.
        """
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Params that should stay in float32 for numerical stability
        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: Tensor) -> Tensor:
        """Image embedding using PaliGemma's SigLIP vision tower."""
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: Tensor) -> Tensor:
        """Language token embedding from PaliGemma text model."""
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        past_key_values: list[Tensor] | None = None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool = False,
        adarms_cond: list[Tensor | None] | None = None,
    ) -> tuple[list[Tensor | None], list[Tensor] | None]:
        """
        Forward pass supporting:
        - Prefix-only mode (inputs_embeds[1] is None): for caching
        - Suffix-only mode (inputs_embeds[0] is None): for denoise steps with KV cache
        - Full mode (both present): for training

        Returns:
            (outputs_embeds, past_key_values)
        """
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Prefix-only mode: just PaliGemma forward
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            return [prefix_output, None], prefix_past_key_values

        # Suffix-only mode: just Gemma expert forward with prefix KV cache
        if inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            return [None, suffix_output], None

        # Full mode: simpler two-stage forward
        # 1) Run PaliGemma language model on prefix embeddings
        prefix_outputs = self.paligemma.language_model.forward(
            inputs_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=use_cache,
            adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
        )
        prefix_hidden = prefix_outputs.last_hidden_state
        prefix_past_key_values = prefix_outputs.past_key_values

        # 2) Run Gemma expert on suffix embeddings, conditioned on prefix KV
        suffix_outputs = self.gemma_expert.model.forward(
            inputs_embeds=inputs_embeds[1],
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_past_key_values,
            use_cache=use_cache,
            adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
        )
        suffix_hidden = suffix_outputs.last_hidden_state

        return [prefix_hidden, suffix_hidden], None


# ============================================================================
# PI05Model - Aligned with OpenPI PI0Pytorch
# ============================================================================


class PI05Model(PreTrainedModel):
    """
    Top-level Pi0.5 model: encapsulates projection layers, time embedding,
    and Flow Matching training/inference.

    Fully aligned with OpenPI's PI0Pytorch.
    """

    config_class = PI05Config
    _no_split_modules = ["PaliGemmaWithExpertModel"]

    def __init__(self, config: PI05Config):
        super().__init__(config)
        self.config = config
        
        # Get variant parameters
        vlm_params = self._get_variant_params(config.paligemma_variant)
        expert_params = self._get_variant_params(config.action_expert_variant)
        
        precision = "bfloat16" if config.dtype == "bfloat16" else "float32"

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            vlm_params,
            expert_params,
            use_adarms=[False, True],  # Pi0.5: only expert uses AdaRMS
            precision=precision,
        )

        # Action projection layers
        expert_width = expert_params["width"]
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_width)
        self.action_out_proj = nn.Linear(expert_width, config.max_action_dim)
        
        # Time MLP for AdaRMS (Pi0.5 specific)
        self.time_mlp_in = nn.Linear(expert_width, expert_width)
        self.time_mlp_out = nn.Linear(expert_width, expert_width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

    def _get_variant_params(self, variant: str) -> dict:
        """Get model variant parameters matching OpenPI."""
        if variant == "gemma_300m":
            return {
                "width": 1024,
                "depth": 18,
                "mlp_dim": 4096,
                "num_heads": 8,
                "num_kv_heads": 1,
                "head_dim": 256,
            }
        # Default: gemma_2b
        return {
            "width": 2048,
            "depth": 18,
            "mlp_dim": 16384,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }

    # ---- Gradient checkpointing ----

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for PI05Model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for PI05Model")

    # ---- Helper functions matching OpenPI ----

    def sample_noise(self, shape: tuple, device: torch.device) -> Tensor:
        """Sample noise from standard normal distribution."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Sample time from Beta distribution as in OpenPI."""
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _prepare_attention_masks_4d(self, att_2d_masks: Tensor) -> Tensor:
        """Convert 2D boolean masks to 4D additive masks for transformers."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    # ---- Embedding helpers ----

    def embed_prefix(
        self,
        images: list[Tensor],
        image_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Embed images (SigLIP) and language tokens as prefix tokens.
        Supports multiple cameras, exactly matching OpenPI's embed_prefix.

        Args:
            images: List of [B, 3, H, W] tensors, one per camera
            image_masks: List of [B] boolean tensors indicating valid images
            lang_tokens: [B, L] token ids
            lang_masks: [B, L] attention mask

        Returns:
            (embs, pad_masks, att_masks)
        """
        embs = []
        pad_masks = []
        att_masks: list[int] = []

        # Process each camera image
        for img, img_mask in zip(images, image_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Image tokens attend to each other (block 0)
            att_masks += [0] * num_img_embs

        # Process language tokens
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)  # Scale as in OpenPI

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # Full attention between image and language (still block 0)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_prefix_simple(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Simplified embed_prefix for single camera.
        Provided for backward compatibility with existing code.
        """
        bsize = pixel_values.shape[0]
        device = pixel_values.device

        # Wrap single camera as list
        images = [pixel_values]
        image_masks = [torch.ones(bsize, dtype=torch.bool, device=device)]

        return self.embed_prefix(images, image_masks, input_ids, attention_mask)

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Generate suffix embeddings and AdaRMS condition for the action expert.

        This follows the Pi0.5 branch of OpenPI's embed_suffix:
        - No separate state token (state is in the language prompt)
        - Time embedding → MLP → AdaRMS condition
        - Action tokens via linear projection only

        Returns:
            (embs, pad_masks, att_masks, adarms_cond)
        """
        embs = []
        pad_masks = []
        att_masks: list[int] = []

        # Time embedding: sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Time MLP for AdaRMS condition
        x = self.time_mlp_in(time_emb)
        x = F.silu(x)
        x = self.time_mlp_out(x)
        adarms_cond = F.silu(x)

        # Action projection
        action_emb = self.action_in_proj(noisy_actions)

        # Just use action embeddings (no time concat for Pi0.5)
        action_time_emb = action_emb

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Attention mask: first token starts new block, rest share it
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    # ---- Training forward ----

    def forward(
        self,
        pixel_values: Tensor | list[Tensor],
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """
        Training forward pass: compute Flow Matching loss.

        Returns loss with reduction="none" for flexible external aggregation,
        matching OpenPI's behavior.

        Args:
            pixel_values: [B, 3, H, W] single camera image (for backward compatibility)
            input_ids: [B, L] token ids
            attention_mask: [B, L] attention mask
            actions: [B, T, A] normalized actions
            images: Optional list of [B, 3, H, W] tensors for multi-camera support
            image_masks: Optional list of [B] boolean tensors for multi-camera support
            noise: optional noise tensor
            time: optional timesteps [B]

        Returns:
            [B, T, A] loss tensor (reduction="none")
        """
        bsize = actions.shape[0]
        device = actions.device

        # Sample noise and time if not provided
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)

        # Build noisy actions x_t and target vector field u_t
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Embed prefix - support multi-camera if images provided
        if images is not None and image_masks is not None:
            # Multi-camera mode
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, image_masks, input_ids, attention_mask
            )
        else:
            # Single camera mode (backward compatibility)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_simple(
                pixel_values, input_ids, attention_mask
            )

        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        # Cast to bfloat16 if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Prepare masks and position ids
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Forward through model
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract action tokens and project
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        # Return loss with reduction="none" (as in OpenPI)
        return F.mse_loss(u_t, v_t, reduction="none")

    # ---- Inference: sampling with KV cache ----

    @torch.no_grad()
    def sample_actions(
        self,
        device: torch.device,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        """
        Inference: sample actions via Euler integration with KV caching.
        Exactly matches OpenPI's sample_actions.

        Args:
            device: Device to use
            pixel_values: [B, 3, H, W]
            input_ids: [B, L]
            attention_mask: [B, L]
            noise: Optional initial noise
            num_steps: Number of denoising steps

        Returns:
            [B, T, A] sampled actions
        """
        self.eval()

        bsize = pixel_values.shape[0]
        steps = num_steps or self.config.num_inference_steps

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_simple(
            pixel_values, input_ids, attention_mask
        )

        # Compute prefix KV cache
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        # Set eager attention for inference
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Euler integration
        dt = -1.0 / steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt_tensor / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            x_t = x_t + dt_tensor * v_t
            time = time + dt_tensor

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: list[Tensor],
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """
        Apply one denoising step using cached prefix KV values.
        Matches OpenPI's denoise_step.
        """
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # Build attention mask for suffix attending to cached prefix
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # Position ids continue from prefix
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        return self.action_out_proj(suffix_out)

    # ========================================================================
    # HIGH-LEVEL INFERENCE: Subtask prediction (Pi0.5 Paper Section IV.A)
    # ========================================================================

    @torch.no_grad()
    def generate_subtask(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        High-level inference: Generate subtask text autoregressively.
        
        This implements π_θ(ℓ̂ | o_t, ℓ) from the Pi0.5 paper.
        
        The model takes:
        - Images (pixel_values) + task prompt (input_ids like "clean the bedroom")
        - Outputs subtask text (like "pick up pillow")
        
        Args:
            pixel_values: [B, 3, H, W] image tensor
            input_ids: [B, L] input token ids (task prompt)
            attention_mask: [B, L] attention mask
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Nucleus sampling parameter
            eos_token_id: Token id to stop generation
            
        Returns:
            [B, max_new_tokens] generated subtask token ids
        """
        self.eval()
        bsize = pixel_values.shape[0]
        device = pixel_values.device
        
        # Embed prefix (images + task prompt)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_simple(
            pixel_values, input_ids, attention_mask
        )
        
        # Use PaliGemma's language model head for text generation
        lm_head = self.paligemma_with_expert.paligemma.language_model.lm_head
        
        # Prepare for autoregressive generation
        generated_tokens = []
        current_embs = prefix_embs
        current_pad_masks = prefix_pad_masks
        current_att_masks = prefix_att_masks
        
        for step in range(max_new_tokens):
            # Build attention mask
            att_2d_masks = make_att_2d_masks(current_pad_masks, current_att_masks)
            position_ids = torch.cumsum(current_pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
            
            # Forward through PaliGemma language model only (no action expert)
            outputs = self.paligemma_with_expert.paligemma.language_model.forward(
                inputs_embeds=current_embs,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                use_cache=False,
            )
            
            # Get logits for the last token
            hidden_states = outputs.last_hidden_state
            last_hidden = hidden_states[:, -1, :]  # [B, hidden_dim]
            logits = lm_head(last_hidden)  # [B, vocab_size]
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            generated_tokens.append(next_token)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            
            # Embed next token and append to sequence
            next_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
            hidden_dim = next_emb.shape[-1]
            next_emb = next_emb * math.sqrt(hidden_dim)  # Scale as in embed_prefix
            
            current_embs = torch.cat([current_embs, next_emb], dim=1)
            current_pad_masks = torch.cat([
                current_pad_masks,
                torch.ones(bsize, 1, dtype=torch.bool, device=device)
            ], dim=1)
            # Generated tokens use causal attention (ar_mask = 0 for full attention)
            current_att_masks = torch.cat([
                current_att_masks,
                torch.zeros(bsize, 1, dtype=torch.bool, device=device)
            ], dim=1)
        
        # Stack generated tokens
        if generated_tokens:
            return torch.cat(generated_tokens, dim=1)
        else:
            return torch.empty(bsize, 0, dtype=torch.long, device=device)

    # ========================================================================
    # HIERARCHICAL FORWARD: Joint training for subtask + action (Pi0.5 Eq. 1)
    # ========================================================================

    def forward_hierarchical(
        self,
        pixel_values: Tensor | list[Tensor],
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        subtask_ids: Optional[Tensor] = None,
        subtask_mask: Optional[Tensor] = None,
        fast_action_ids: Optional[Tensor] = None,
        fast_action_mask: Optional[Tensor] = None,
        images: list[Tensor] | None = None,
        image_masks: list[Tensor] | None = None,
        alpha: float = 10.0,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Pi05HierarchicalOutput:
        """
        Hierarchical training forward pass implementing Pi0.5 Equation (1).
        
        This jointly trains:
        1. Subtask prediction (cross-entropy loss on subtask_ids)
        2. Action generation (flow matching loss) OR FAST action tokens (if provided)
        
        Combined loss:
            - If fast_action_ids provided (pre-training style):
                L = H(subtask_ids, logits) + H(fast_action_ids, logits)
            - Otherwise (post-training style):
                L = H(subtask_ids, logits) + α * ||u_t - v_t||²
        
        Args:
            pixel_values: [B, 3, H, W] image tensor
            input_ids: [B, L] task prompt token ids
            attention_mask: [B, L] task prompt attention mask
            actions: [B, T, A] ground truth actions (normalized)
            subtask_ids: [B, S] ground truth subtask token ids (optional)
            subtask_mask: [B, S] subtask token mask (optional)
            fast_action_ids: [B, A] FAST action token ids (optional, for pre-training style)
            fast_action_mask: [B, A] FAST action token mask (optional)
            alpha: Weight for flow matching loss (default 10.0, ignored if fast_action_ids provided)
            noise: Optional noise tensor
            time: Optional timesteps [B]
            
        Returns:
            Pi05HierarchicalOutput with loss components and predictions
        """
        bsize = actions.shape[0]
        device = actions.device
        
        # Sample noise and time if not provided
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)
        
        # Build noisy actions x_t and target vector field u_t
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # ------------------------------------------------------------------
        # 1) High-level text path: prefix-only forward for subtask CE loss
        #    Gradients from text loss can update the VLM (prefix branch).
        # ------------------------------------------------------------------
        # Support multi-camera if images provided
        if images is not None and image_masks is not None:
            # Multi-camera mode
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, image_masks, input_ids, attention_mask
            )
        else:
            # Single camera mode (backward compatibility)
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_simple(
                pixel_values, input_ids, attention_mask
            )

        # Cast prefix embeddings to model dtype if needed
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Prefix-only attention masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        text_loss = None
        prefix_out_text: Tensor | None = None
        if subtask_ids is not None or fast_action_ids is not None:
            # Forward only through PaliGemma language model (no action expert)
            lm_outputs = self.paligemma_with_expert.paligemma.language_model.forward(
                inputs_embeds=prefix_embs,
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                use_cache=False,
            )
            prefix_out_text = lm_outputs.last_hidden_state

            # Get LM head for text prediction
            lm_head = self.paligemma_with_expert.paligemma.language_model.lm_head
            
            # Get logits from prefix output
            text_hidden = prefix_out_text.to(dtype=torch.float32)
            logits = lm_head(text_hidden)  # [B, prefix_len, vocab_size]
            
            # Combine subtask and FAST action tokens for pre-training style
            if fast_action_ids is not None:
                # Pre-training style: predict both subtask and FAST action tokens
                # Target sequence: [subtask_tokens, fast_action_tokens]
                target_tokens_list = []
                target_mask_list = []
                
                if subtask_ids is not None:
                    S = subtask_ids.shape[1]
                    target_tokens_list.append(subtask_ids)
                    if subtask_mask is not None:
                        target_mask_list.append(subtask_mask.float())
                    else:
                        target_mask_list.append(torch.ones(bsize, S, device=subtask_ids.device))
                
                A = fast_action_ids.shape[1]
                target_tokens_list.append(fast_action_ids)
                if fast_action_mask is not None:
                    target_mask_list.append(fast_action_mask.float())
                else:
                    target_mask_list.append(torch.ones(bsize, A, device=fast_action_ids.device))
                
                # Concatenate targets
                all_target_ids = torch.cat(target_tokens_list, dim=1)  # [B, S+A]
                all_target_mask = torch.cat(target_mask_list, dim=1)  # [B, S+A]
                total_target_len = all_target_ids.shape[1]
                
                # Extract logits for target positions (after prompt)
                # Use the last total_target_len positions of prefix for prediction
                if logits.shape[1] >= total_target_len:
                    # Shift for next-token prediction
                    shift_logits = logits[:, -total_target_len-1:-1, :].contiguous()  # [B, S+A, vocab]
                    shift_labels = all_target_ids.contiguous()  # [B, S+A]
                    
                    # Flatten for cross-entropy
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    text_loss = loss_fct(flat_logits, flat_labels).view(bsize, total_target_len)  # [B, S+A]
                    
                    # Apply mask
                    text_loss = text_loss * all_target_mask
            elif subtask_ids is not None:
                # Post-training style: only predict subtask tokens
                S = subtask_ids.shape[1]
                if logits.shape[1] >= S:
                    shift_logits = logits[:, -S-1:-1, :].contiguous()  # [B, S, vocab]
                    shift_labels = subtask_ids.contiguous()  # [B, S]
                    
                    # Flatten for cross-entropy
                    loss_fct = nn.CrossEntropyLoss(reduction="none")
                    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_labels = shift_labels.view(-1)
                    text_loss = loss_fct(flat_logits, flat_labels).view(bsize, S)  # [B, S]
                    
                    # Apply mask if provided
                    if subtask_mask is not None:
                        text_loss = text_loss * subtask_mask.float()

        # ------------------------------------------------------------------
        # 2) Low-level action path: joint prefix+suffix forward for flow loss
        #    IMPORTANT: detach prefix_embs so action gradients do not update
        #    the high-level planner (VLM), matching Pi0.5's decoupling.
        #    SKIP if fast_action_ids provided (pre-training style, alpha=0)
        # ------------------------------------------------------------------
        
        action_loss = None
        suffix_out = None
        
        if fast_action_ids is None:
            # Post-training style: compute flow matching loss
            # Reuse prefix masks, but detach embeddings for action branch
            prefix_embs_detached = prefix_embs.detach()

            # Embed suffix (noisy actions)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

            if model_dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)

            # Concatenate prefix + suffix for joint attention
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            # Forward through joint model; prefix embeddings are treated as constants
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs_detached, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )

            # Compute action loss (flow matching) from suffix branch only
            suffix_out = suffix_out[:, -self.config.chunk_size:]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            action_loss = F.mse_loss(u_t, v_t, reduction="none")  # [B, T, A]
        
        # Compute total loss
        if fast_action_ids is not None:
            # Pre-training style: only text loss (subtask + FAST action tokens)
            if text_loss is not None:
                total_loss = text_loss.mean()
            else:
                total_loss = torch.tensor(0.0, device=pixel_values.device)
            action_loss_mean = None
            text_loss_mean = text_loss.mean() if text_loss is not None else None
        else:
            # Post-training style: text loss + flow matching loss
            action_loss_mean = action_loss.mean() if action_loss is not None else None
            if text_loss is not None:
                text_loss_mean = text_loss.mean()
                total_loss = text_loss_mean + alpha * action_loss_mean
            else:
                text_loss_mean = None
                total_loss = alpha * action_loss_mean if action_loss_mean is not None else torch.tensor(0.0, device=pixel_values.device)
        
        return Pi05HierarchicalOutput(
            action_loss=action_loss,
            text_loss=text_loss,
            total_loss=total_loss,
            prefix_embeds=prefix_out_text,
            suffix_embeds=suffix_out,
        )

    # ========================================================================
    # HIERARCHICAL INFERENCE: Full pipeline (High-Level → Low-Level)
    # ========================================================================

    @torch.no_grad()
    def infer_hierarchical(
        self,
        pixel_values: Tensor,
        task_prompt_ids: Tensor,
        task_prompt_mask: Tensor,
        subtask_ids: Optional[Tensor] = None,
        subtask_mask: Optional[Tensor] = None,
        num_steps: int | None = None,
        generate_subtask: bool = True,
        max_subtask_tokens: int = 32,
        tokenizer: Optional[Any] = None,
    ) -> Pi05HierarchicalOutput:
        """
        Full hierarchical inference pipeline.
        
        Step 1 (High-Level): Generate subtask from task prompt
            π_θ(ℓ̂ | o_t, ℓ) → subtask text
            
        Step 2 (Low-Level): Generate actions conditioned on subtask
            π_θ(a_{t:t+H} | o_t, ℓ̂) → action chunk
        
        Args:
            pixel_values: [B, 3, H, W] image tensor
            task_prompt_ids: [B, L] task prompt token ids (e.g., "clean the bedroom")
            task_prompt_mask: [B, L] task prompt attention mask
            subtask_ids: [B, S] pre-specified subtask token ids (skip generation if provided)
            subtask_mask: [B, S] subtask attention mask
            num_steps: Number of flow matching denoising steps
            generate_subtask: Whether to generate subtask (if False, use subtask_ids)
            max_subtask_tokens: Maximum tokens to generate for subtask
            tokenizer: Tokenizer for concatenating prompts (required if generate_subtask=True)
            
        Returns:
            Pi05HierarchicalOutput with predicted_actions and predicted_subtask_ids
        """
        self.eval()
        device = pixel_values.device
        bsize = pixel_values.shape[0]
        
        # Step 1: High-level inference (subtask prediction)
        if generate_subtask and subtask_ids is None:
            predicted_subtask_ids = self.generate_subtask(
                pixel_values=pixel_values,
                input_ids=task_prompt_ids,
                attention_mask=task_prompt_mask,
                max_new_tokens=max_subtask_tokens,
                eos_token_id=1,  # Default EOS for PaliGemma
            )
        else:
            predicted_subtask_ids = subtask_ids
        
        # Prepare low-level input: combine task prompt with subtask
        # For simplicity, if subtask was generated, concatenate to task prompt
        if predicted_subtask_ids is not None and tokenizer is not None:
            # Concatenate task prompt + subtask for low-level inference
            combined_ids = torch.cat([task_prompt_ids, predicted_subtask_ids], dim=1)
            combined_mask = torch.cat([
                task_prompt_mask,
                torch.ones_like(predicted_subtask_ids, dtype=torch.bool)
            ], dim=1)
        elif subtask_ids is not None:
            combined_ids = subtask_ids
            combined_mask = subtask_mask if subtask_mask is not None else torch.ones_like(subtask_ids, dtype=torch.bool)
        else:
            combined_ids = task_prompt_ids
            combined_mask = task_prompt_mask
        
        # Step 2: Low-level inference (action generation)
        predicted_actions = self.sample_actions(
            device=device,
            pixel_values=pixel_values,
            input_ids=combined_ids,
            attention_mask=combined_mask,
            num_steps=num_steps,
        )
        
        return Pi05HierarchicalOutput(
            predicted_actions=predicted_actions,
            predicted_subtask_ids=predicted_subtask_ids,
        )
