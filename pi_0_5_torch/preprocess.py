"""
Pi0.5 Preprocessing - Fully aligned with OpenPI official implementation.

This module implements preprocessing that matches:
1. Multi-camera image processing with augmentation (from preprocessing_pytorch.py)
2. State normalization and discretization
3. Prompt construction with discretized state
4. Action normalization/denormalization
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoTokenizer, SiglipImageProcessor

from .configuration import PI05Config

logger = logging.getLogger(__name__)

# Default image keys matching OpenPI
DEFAULT_IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


# ============================================================================
# Helper functions
# ============================================================================


def pad_vector(vec: Tensor, target_dim: int) -> Tensor:
    """Pad or truncate vector to target dimension (matches OpenPI's pad_vector)."""
    d = vec.shape[-1]
    if d == target_dim:
        return vec
    if d > target_dim:
        return vec[..., :target_dim]
    pad = torch.zeros(
        (*vec.shape[:-1], target_dim - d),
        dtype=vec.dtype,
        device=vec.device,
    )
    return torch.cat([vec, pad], dim=-1)


def discretize_state_256(state_norm: Tensor) -> Tensor:
    """
    Discretize normalized [-1, 1] state to 0..255 integers.

    Matches the logic in OpenPI's processor_pi05.py:
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    """
    state_clipped = state_norm.clamp(-1.0, 1.0)
    # Map [-1, 1] to [0, 256) then floor
    bins = 256
    scaled = (state_clipped + 1.0) * (bins / 2.0)
    indices = torch.clamp(scaled.floor(), 0, bins - 1).long()
    return indices


def normalize_feature(x: Tensor, stats: Optional[Dict[str, Tensor]]) -> Tensor:
    """
    Normalize feature using mean/std stats.

    For compatibility with OpenPI's NormalizerProcessorStep, stats structure:
        {"mean": Tensor[dim], "std": Tensor[dim]}

    If stats is None or incomplete, returns original tensor.
    """
    if stats is None:
        return x
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return x

    # Broadcast to batch dimension
    while mean.dim() < x.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return (x - mean.to(x.device)) / (std.to(x.device) + 1e-6)


def denormalize_feature(x: Tensor, stats: Optional[Dict[str, Tensor]]) -> Tensor:
    """Inverse of normalize_feature."""
    if stats is None:
        return x
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return x

    while mean.dim() < x.dim():
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return x * std.to(x.device) + mean.to(x.device)


# ============================================================================
# Image augmentation - Aligned with OpenPI preprocessing_pytorch.py
# ============================================================================


def apply_image_augmentation(
    image: Tensor,
    camera_key: str,
    train: bool = True,
) -> Tensor:
    """
    Apply image augmentation as in OpenPI's preprocess_observation_pytorch.

    Args:
        image: [B, H, W, C] tensor in range [-1, 1]
        camera_key: Camera name for determining augmentation type
        train: Whether in training mode

    Returns:
        Augmented image tensor
    """
    if not train:
        return image

    # Convert from [-1, 1] to [0, 1] for augmentation
    image = image / 2.0 + 0.5

    height, width = image.shape[1:3]

    # Geometric augmentations for non-wrist cameras only
    if "wrist" not in camera_key:
        # Random crop and resize
        crop_height = int(height * 0.95)
        crop_width = int(width * 0.95)

        max_h = height - crop_height
        max_w = width - crop_width
        if max_h > 0 and max_w > 0:
            start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
            start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
            image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

        # Resize back
        image = torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),  # [B, H, W, C] -> [B, C, H, W]
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # Random rotation (small angles)
        angle = torch.rand(1, device=image.device) * 10 - 5  # -5 to 5 degrees
        if torch.abs(angle) > 0.1:
            angle_rad = angle * torch.pi / 180.0
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)

            grid_x = torch.linspace(-1, 1, width, device=image.device)
            grid_y = torch.linspace(-1, 1, height, device=image.device)
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

            grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
            grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

            grid_x_rot = grid_x * cos_a - grid_y * sin_a
            grid_y_rot = grid_x * sin_a + grid_y * cos_a

            grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

            image = torch.nn.functional.grid_sample(
                image.permute(0, 3, 1, 2),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).permute(0, 2, 3, 1)

    # Color augmentations for all cameras
    # Random brightness
    brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # 0.7 to 1.3
    image = image * brightness_factor

    # Random contrast
    contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # 0.6 to 1.4
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast_factor + mean

    # Random saturation
    saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # 0.5 to 1.5
    gray = image.mean(dim=-1, keepdim=True)
    image = gray + (image - gray) * saturation_factor

    # Clamp and convert back to [-1, 1]
    image = torch.clamp(image, 0, 1)
    image = image * 2.0 - 1.0

    return image


def resize_with_pad(image: Tensor, target_height: int, target_width: int) -> Tensor:
    """
    Resize image with padding to maintain aspect ratio.
    Simplified version matching OpenPI's resize_with_pad_torch.
    """
    # For now, just use bilinear interpolation
    # Input: [B, H, W, C]
    image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
    image = torch.nn.functional.interpolate(
        image,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )
    return image.permute(0, 2, 3, 1)  # [B, H, W, C]


# ============================================================================
# Dataset statistics container
# ============================================================================


@dataclass
class Pi05DatasetStats:
    """
    Container for state/action statistics used in normalization.

    For full OpenPI compatibility, these should be computed from the dataset
    using quantile normalization. This implementation supports mean/std
    as a simpler alternative.
    """

    state_stats: Optional[Dict[str, Tensor]] = None  # {"mean": ..., "std": ...}
    action_stats: Optional[Dict[str, Tensor]] = None


# ============================================================================
# Main Preprocessor class
# ============================================================================


class Pi05Preprocessor:
    """
    Preprocessing pipeline aligned with OpenPI's processor_pi05.py.

    Features:
    - Multi-camera image processing with augmentation
    - State padding and normalization
    - State discretization to 256 bins
    - Prompt construction with discretized state
    - Action normalization
    - Tokenization with PaliGemma tokenizer
    """

    def __init__(
        self,
        config: PI05Config,
        dataset_stats: Optional[Pi05DatasetStats] = None,
        device: Optional[str] = None,
        image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
        use_augmentation: bool = True,
        fast_tokenizer_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_stats = dataset_stats or Pi05DatasetStats()
        self.image_keys = image_keys
        self.use_augmentation = use_augmentation

        # Tokenizer and image processor matching OpenPI
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            padding_side="right",
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(
            "google/paligemma-3b-pt-224",
        )
        
        # FAST tokenizer for discrete action tokenization (optional)
        self.fast_tokenizer = None
        self.fast_to_vlm_mapping: Optional[Dict[int, int]] = None
        if fast_tokenizer_path is not None:
            from transformers import AutoProcessor
            import pickle
            import os
            
            logger.info(f"Loading FAST tokenizer from {fast_tokenizer_path}")
            self.fast_tokenizer = AutoProcessor.from_pretrained(
                fast_tokenizer_path,
                trust_remote_code=True,
            )
            
            # Try to load action_to_vlm mapping if available
            # This mapping is created by domain_transfer.py
            mapping_path = os.path.join(fast_tokenizer_path, "..", "domain_transfer_info.pkl")
            if not os.path.exists(mapping_path):
                mapping_path = os.path.join(fast_tokenizer_path, "domain_transfer_info.pkl")
            
            if os.path.exists(mapping_path):
                try:
                    with open(mapping_path, "rb") as f:
                        info = pickle.load(f)
                    self.fast_to_vlm_mapping = info.get("action_to_vlm_mapping")
                    if self.fast_to_vlm_mapping:
                        logger.info(f"Loaded FAST→VLM token mapping ({len(self.fast_to_vlm_mapping)} tokens)")
                except Exception as e:
                    logger.warning(f"Could not load FAST→VLM mapping: {e}")
            else:
                logger.warning(
                    f"FAST→VLM mapping not found at {mapping_path}. "
                    "FAST token IDs will be used directly (assumes VLM vocab already extended)."
                )

    def _process_single_image(self, image: Image.Image) -> Tensor:
        """Process a single PIL image to tensor in [-1, 1] range."""
        img_inputs = self.image_processor(
            image,
            return_tensors="pt",
            do_resize=True,
            size={
                "height": self.config.image_resolution[0],
                "width": self.config.image_resolution[1],
            },
        )
        # Scale to [-1, 1]
        pixel_values = img_inputs.pixel_values * 2.0 - 1.0
        return pixel_values.squeeze(0)  # [3, H, W]

    def _process_images_batch(
        self,
        images: List[Image.Image] | Dict[str, List[Image.Image]],
        train: bool = True,
    ) -> tuple[List[Tensor], List[Tensor]]:
        """
        Process images for a batch.

        Args:
            images: Either a list of PIL images (single camera) or
                   dict mapping camera_key -> list of PIL images (multi-camera)
            train: Whether to apply augmentation

        Returns:
            (list of [B, 3, H, W] tensors, list of [B] mask tensors)
        """
        if isinstance(images, dict):
            # Multi-camera case
            # Check that all required image keys are present (matching OpenPI behavior)
            missing_keys = set(self.image_keys) - set(images.keys())
            if missing_keys:
                raise ValueError(
                    f"Multi-camera images dict missing required keys: expected {self.image_keys}, "
                    f"got {list(images.keys())}, missing {list(missing_keys)}"
                )
            
            processed_images = []
            image_masks = []

            # Process images in the order specified by image_keys (matching OpenPI)
            for key in self.image_keys:
                cam_images = images[key]
                batch_tensors = [self._process_single_image(img) for img in cam_images]
                batch_tensor = torch.stack(batch_tensors, dim=0)  # [B, 3, H, W]

                # Apply augmentation if training
                if train and self.use_augmentation:
                    # Convert to [B, H, W, C] for augmentation
                    batch_tensor = batch_tensor.permute(0, 2, 3, 1)
                    batch_tensor = apply_image_augmentation(batch_tensor, key, train=True)
                    batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # Back to [B, C, H, W]

                processed_images.append(batch_tensor)
                # All images are valid (mask = True)
                image_masks.append(torch.ones(len(cam_images), dtype=torch.bool))

            return processed_images, image_masks
        else:
            # Single camera case (backward compatibility)
            batch_tensors = [self._process_single_image(img) for img in images]
            batch_tensor = torch.stack(batch_tensors, dim=0)  # [B, 3, H, W]

            if train and self.use_augmentation:
                batch_tensor = batch_tensor.permute(0, 2, 3, 1)
                batch_tensor = apply_image_augmentation(batch_tensor, "single_camera", train=True)
                batch_tensor = batch_tensor.permute(0, 3, 1, 2)

            return [batch_tensor], [torch.ones(len(images), dtype=torch.bool)]

    def _build_prompts(self, texts: List[str], state_norm: Tensor) -> List[str]:
        """
        Build prompts with discretized state.

        Format: "Task: <cleaned_text>, State: <idx idx idx ...>;\nAction: "
        """
        with torch.no_grad():
            state_idx = discretize_state_256(state_norm.cpu())  # [B, D]

        B = state_idx.shape[0]
        prompts: List[str] = []

        for i in range(B):
            text = texts[i] or ""
            cleaned_text = text.strip().replace("_", " ").replace("\n", " ").replace("\r", " ")

            idx_list = state_idx[i].tolist()
            state_str = " ".join(str(int(x)) for x in idx_list)

            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            prompts.append(full_prompt)

        return prompts

    def _build_hierarchical_prompts(
        self,
        high_level_tasks: List[str],
        subtasks: List[str],
        state_norm: Tensor,
        mode: str = "high_level",
        pre_training_style: bool = False,
    ) -> tuple[List[str], List[str]]:
        """
        Build prompts for hierarchical policy training (Pi0.5 paper).
        
        High-level mode (subtask prediction):
            - Post-training style: "Task: <high_level_task>, State: <state>;\nSubtask: "
              Target: <subtask> (only subtask tokens)
            - Pre-training style: "Task: <high_level_task>, State: <state>;\n"
              Target: <subtask> + <fast_action_tokens> (continuous sequence)
            
        Low-level mode (action generation):
            Input: "Task: <subtask>, State: <state>;\nAction: "
            Target: actions
        
        Args:
            high_level_tasks: List of high-level task descriptions
            subtasks: List of subtask labels
            state_norm: [B, D] normalized state tensor
            mode: "high_level" or "low_level"
            pre_training_style: If True, use pre-training prompt format (no "Subtask: " suffix)
                                to predict [subtask, fast_action_tokens] as continuous sequence
            
        Returns:
            (prompts, targets) where targets are subtask texts for high-level mode
        """
        with torch.no_grad():
            state_idx = discretize_state_256(state_norm.cpu())

        B = state_idx.shape[0]
        prompts: List[str] = []
        targets: List[str] = []

        for i in range(B):
            idx_list = state_idx[i].tolist()
            state_str = " ".join(str(int(x)) for x in idx_list)
            
            if mode == "high_level":
                # High-level inference: predict subtask from task
                hl_task = (high_level_tasks[i] or "").strip().replace("_", " ").replace("\n", " ")
                if pre_training_style:
                    # Pre-training style: predict [subtask, fast_action_tokens] as continuous sequence
                    # No "Subtask: " suffix, model naturally predicts the full sequence
                    prompt = f"Task: {hl_task}, State: {state_str};\n"
                else:
                    # Post-training style: only predict subtask tokens
                    prompt = f"Task: {hl_task}, State: {state_str};\nSubtask: "
                target = (subtasks[i] or "").strip().replace("_", " ").replace("\n", " ")
            else:
                # Low-level inference: predict actions from subtask
                subtask = (subtasks[i] or "").strip().replace("_", " ").replace("\n", " ")
                prompt = f"Task: {subtask}, State: {state_str};\nAction: "
                target = ""  # Actions are the target, not text
            
            prompts.append(prompt)
            targets.append(target)

        return prompts, targets

    def preprocess_batch(
        self,
        batch: Dict[str, Any],
        train: bool = True,
    ) -> Dict[str, Any]:
        """
        Preprocess a batch for Pi0.5 model.

        Input batch format (from Pi05Dataset + pi05_collate_fn):
            - images: List[PIL.Image] or Dict[str, List[PIL.Image]] (multi-camera)
            - states: Tensor [B, state_dim]
            - actions: Tensor [B, T, act_dim]
            - action_is_pad: Tensor [B, T]
            - texts: List[str]

        Output:
            - pixel_values: Tensor [B, 3, H, W] or List[Tensor] (multi-camera)
            - image_masks: List[Tensor [B]] (multi-camera)
            - input_ids: Tensor [B, L]
            - attention_mask: Tensor [B, L]
            - actions: Tensor [B, T, act_dim] (normalized)
            - action_is_pad: Tensor [B, T]
            - state_continuous: Tensor [B, max_state_dim]
            - state_normalized: Tensor [B, max_state_dim]
            - raw_texts: List[str]
        """
        images = batch["images"]
        states: Tensor = batch["states"]
        actions: Tensor = batch["actions"]
        action_is_pad: Tensor = batch["action_is_pad"]
        texts: List[str] = batch["texts"]

        B = states.shape[0]

        # 1. Process images
        processed_images, image_masks = self._process_images_batch(images, train=train)

        # 2. Pad state to max_state_dim
        states_padded = pad_vector(states, self.config.max_state_dim)

        # 3. Normalize state
        state_norm = normalize_feature(states_padded, self.dataset_stats.state_stats)

        # 4. Build prompts with discretized state
        prompts = self._build_prompts(texts, state_norm)

        # 5. Tokenize
        tok_outputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
        )
        input_ids = tok_outputs.input_ids
        attention_mask = tok_outputs.attention_mask

        # 6. Normalize actions
        actions_norm = normalize_feature(actions, self.dataset_stats.action_stats)

        # Move to device
        processed_images = [img.to(self.device) for img in processed_images]
        image_masks = [mask.to(self.device) for mask in image_masks]
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        actions_norm = actions_norm.to(self.device)
        action_is_pad = action_is_pad.to(self.device)
        states_padded = states_padded.to(self.device)
        state_norm = state_norm.to(self.device)

        # For backward compatibility, also provide single pixel_values
        # Always use first camera image (for backward compatibility with single-camera code)
        pixel_values = processed_images[0]

        return {
            "pixel_values": pixel_values,
            "images": processed_images,
            "image_masks": image_masks,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": actions_norm,
            "action_is_pad": action_is_pad,
            "state_continuous": states_padded,
            "state_normalized": state_norm,
            "raw_texts": texts,
        }

    def postprocess_actions(self, actions_pred: Tensor) -> Tensor:
        """
        Denormalize predicted actions back to original scale.

        Args:
            actions_pred: [B, T, act_dim] normalized predictions

        Returns:
            [B, T, act_dim] denormalized actions
        """
        return denormalize_feature(actions_pred, self.dataset_stats.action_stats)

    def preprocess_hierarchical_batch(
        self,
        batch: Dict[str, Any],
        train: bool = True,
    ) -> Dict[str, Any]:
        """
        Preprocess a batch for Pi0.5 hierarchical policy training.
        
        This method handles both high-level and low-level training:
        - High-level: Task prompt → Subtask prediction (cross-entropy)
        - Low-level: Subtask → Action generation (flow matching)
        
        Input batch format:
            - images: List[PIL.Image] or Dict[str, List[PIL.Image]]
            - states: Tensor [B, state_dim]
            - actions: Tensor [B, T, act_dim]
            - action_is_pad: Tensor [B, T]
            - high_level_tasks: List[str]  # High-level task descriptions
            - subtasks: List[str]          # Subtask labels
            
        Output:
            - pixel_values: Tensor [B, 3, H, W]
            - high_level_input_ids: Tensor [B, L] (for high-level inference)
            - high_level_attention_mask: Tensor [B, L]
            - subtask_input_ids: Tensor [B, S] (ground truth subtask tokens)
            - subtask_attention_mask: Tensor [B, S]
            - low_level_input_ids: Tensor [B, L'] (for low-level inference)
            - low_level_attention_mask: Tensor [B, L']
            - actions: Tensor [B, T, act_dim] (normalized)
            - action_is_pad: Tensor [B, T]
        """
        images = batch["images"]
        states: Tensor = batch["states"]
        actions: Tensor = batch["actions"]
        action_is_pad: Tensor = batch["action_is_pad"]
        high_level_tasks: List[str] = batch.get("high_level_tasks", [""] * len(images))
        subtasks: List[str] = batch.get("subtasks", batch.get("texts", [""] * len(images)))
        
        B = states.shape[0]
        
        # 1. Process images
        processed_images, image_masks = self._process_images_batch(images, train=train)
        
        # 2. Pad and normalize state
        states_padded = pad_vector(states, self.config.max_state_dim)
        state_norm = normalize_feature(states_padded, self.dataset_stats.state_stats)
        
        # 3. Build high-level prompts (for subtask prediction)
        # Use pre-training style if FAST tokenizer is enabled (predicts [subtask, fast_action_tokens])
        use_pre_training_style = self.fast_tokenizer is not None
        hl_prompts, subtask_targets = self._build_hierarchical_prompts(
            high_level_tasks, subtasks, state_norm, mode="high_level", pre_training_style=use_pre_training_style
        )
        
        # 4. Build low-level prompts (for action generation)
        ll_prompts, _ = self._build_hierarchical_prompts(
            high_level_tasks, subtasks, state_norm, mode="low_level"
        )
        
        # 5. Tokenize high-level prompts
        hl_tok = self.tokenizer(
            hl_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
        )
        
        # 6. Tokenize subtask targets (for computing cross-entropy loss)
        subtask_tok = self.tokenizer(
            subtask_targets,
            return_tensors="pt",
            padding="max_length",
            max_length=64,  # Subtasks are typically short
            truncation=True,
        )
        
        # 7. Tokenize low-level prompts
        ll_tok = self.tokenizer(
            ll_prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.config.tokenizer_max_length,
            truncation=True,
        )
        
        # 8. Normalize actions
        actions_norm = normalize_feature(actions, self.dataset_stats.action_stats)
        
        # 9. FAST tokenization (if enabled for pre-training style)
        fast_action_ids = None
        fast_action_mask = None
        if self.fast_tokenizer is not None:
            # Convert actions to numpy for FAST tokenizer
            actions_np = actions_norm.detach().cpu().numpy()  # [B, T, A]
            
            fast_token_ids_list = []
            fast_mask_list = []
            
            for b in range(B):
                # Get non-padded action chunk
                valid_len = (~action_is_pad[b]).sum().item()
                if valid_len > 0:
                    action_chunk = actions_np[b, :valid_len]  # [valid_len, A]
                    # FAST tokenizer expects [chunk_size, action_dim]
                    # Pad to chunk_size if needed
                    if valid_len < self.config.chunk_size:
                        padded_chunk = np.zeros((self.config.chunk_size, action_chunk.shape[-1]), dtype=action_chunk.dtype)
                        padded_chunk[:valid_len] = action_chunk
                        action_chunk = padded_chunk
                    
                    # Tokenize action chunk
                    # FAST tokenizer.tokenize() returns token IDs (FAST vocab IDs)
                    try:
                        fast_tokens = self.fast_tokenizer.tokenize(action_chunk)
                        # Convert to list if needed
                        if isinstance(fast_tokens, np.ndarray):
                            fast_tokens = fast_tokens.tolist()
                        elif not isinstance(fast_tokens, list):
                            fast_tokens = [fast_tokens]
                        
                        # Map FAST token IDs to VLM token IDs if mapping available
                        if self.fast_to_vlm_mapping is not None:
                            vlm_tokens = [
                                self.fast_to_vlm_mapping.get(tok_id, tok_id)
                                for tok_id in fast_tokens
                            ]
                            fast_tokens = vlm_tokens
                        
                        fast_token_ids_list.append(fast_tokens)
                        # Create mask (all tokens are valid for non-padded chunks)
                        fast_mask_list.append([1] * len(fast_tokens))
                    except Exception as e:
                        logger.warning(f"FAST tokenization failed for sample {b}: {e}")
                        # Fallback: empty tokens
                        fast_token_ids_list.append([])
                        fast_mask_list.append([])
                else:
                    # All padded, no tokens
                    fast_token_ids_list.append([])
                    fast_mask_list.append([])
            
            # Pad to same length and convert to tensor
            max_fast_len = max(len(tokens) for tokens in fast_token_ids_list) if fast_token_ids_list else 0
            if max_fast_len > 0:
                # Pad all sequences to max_fast_len
                padded_fast_ids = []
                padded_fast_masks = []
                for tokens, mask in zip(fast_token_ids_list, fast_mask_list):
                    pad_len = max_fast_len - len(tokens)
                    padded_tokens = tokens + [0] * pad_len  # 0 is typically PAD token
                    padded_mask = mask + [0] * pad_len
                    padded_fast_ids.append(padded_tokens)
                    padded_fast_masks.append(padded_mask)
                
                fast_action_ids = torch.tensor(padded_fast_ids, dtype=torch.long, device=self.device)
                fast_action_mask = torch.tensor(padded_fast_masks, dtype=torch.bool, device=self.device)
            else:
                # No valid tokens, create empty tensors
                fast_action_ids = torch.zeros((B, 0), dtype=torch.long, device=self.device)
                fast_action_mask = torch.zeros((B, 0), dtype=torch.bool, device=self.device)
        
        # Move to device
        processed_images = [img.to(self.device) for img in processed_images]
        image_masks = [mask.to(self.device) for mask in image_masks]
        
        result = {
            # Images
            # pixel_values: single tensor for backward compatibility (uses first camera)
            "pixel_values": processed_images[0],
            "images": processed_images,
            "image_masks": image_masks,
            
            # High-level inference (subtask prediction)
            "high_level_input_ids": hl_tok.input_ids.to(self.device),
            "high_level_attention_mask": hl_tok.attention_mask.to(self.device),
            
            # Subtask ground truth (for cross-entropy loss)
            "subtask_input_ids": subtask_tok.input_ids.to(self.device),
            "subtask_attention_mask": subtask_tok.attention_mask.to(self.device),
            
            # Low-level inference (action generation)
            "low_level_input_ids": ll_tok.input_ids.to(self.device),
            "low_level_attention_mask": ll_tok.attention_mask.to(self.device),
            
            # Standard fields (backward compatible)
            "input_ids": ll_tok.input_ids.to(self.device),
            "attention_mask": ll_tok.attention_mask.to(self.device),
            
            # Actions
            "actions": actions_norm.to(self.device),
            "action_is_pad": action_is_pad.to(self.device),
            
            # State
            "state_continuous": states_padded.to(self.device),
            "state_normalized": state_norm.to(self.device),
            
            # Raw texts
            "high_level_tasks": high_level_tasks,
            "subtasks": subtasks,
        }
        
        # Add FAST action tokens if available
        if fast_action_ids is not None:
            result["fast_action_ids"] = fast_action_ids
            result["fast_action_mask"] = fast_action_mask
        
        return result
