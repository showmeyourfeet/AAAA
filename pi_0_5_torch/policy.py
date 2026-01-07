"""
Pi0.5 Policy - Fully aligned with OpenPI official implementation.

This module provides a high-level Policy interface similar to OpenPI's Policy class,
handling:
1. Input preprocessing (transforms)
2. Model inference with sample_actions
3. Output postprocessing (denormalization)
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .configuration import PI05Config
from .model import PI05Model
from .preprocess import Pi05DatasetStats, Pi05Preprocessor, pad_vector, normalize_feature, discretize_state_256


class PI05Policy:
    """
    Pi0.5 Policy class aligned with OpenPI's Policy.

    Handles the full inference pipeline:
    1. Preprocess observations (images, state, text)
    2. Run model.sample_actions()
    3. Postprocess actions (denormalization)
    """

    def __init__(
        self,
        config: PI05Config,
        dataset_stats: Optional[Pi05DatasetStats] = None,
        device: Optional[str] = None,
        use_augmentation: bool = False,  # Typically False for inference
    ):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = PI05Model(config)
        self.model.to(self.device)
        self.model.eval()

        # Initialize preprocessor
        self.preprocessor = Pi05Preprocessor(
            config=config,
            dataset_stats=dataset_stats,
            device=self.device,
            use_augmentation=use_augmentation,
        )

        self.dataset_stats = dataset_stats or Pi05DatasetStats()

    def to(self, device: str) -> "PI05Policy":
        """Move model to specified device."""
        self.device = device
        self.model.to(device)
        self.preprocessor.device = device
        return self

    def eval(self) -> "PI05Policy":
        """Set model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "PI05Policy":
        """Set model to training mode."""
        self.model.train(mode)
        return self

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[PI05Config] = None,
        dataset_stats: Optional[Pi05DatasetStats] = None,
        device: Optional[str] = None,
    ) -> "PI05Policy":
        """
        Load policy from checkpoint.

        Args:
            checkpoint_path: Path to safetensors or pt file
            config: Optional config override
            dataset_stats: Optional dataset statistics for normalization
            device: Device to load to

        Returns:
            Loaded PI05Policy
        """
        config = config or PI05Config()
        policy = cls(config=config, dataset_stats=dataset_stats, device=device)

        # Load weights
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle potential key remapping
        new_dict = {}
        for k, v in state_dict.items():
            # Example remapping (adjust based on actual checkpoint format)
            new_k = k.replace("action_time_mlp", "time_mlp")
            new_dict[new_k] = v

        policy.model.load_state_dict(new_dict, strict=False)
        return policy

    def _prepare_single_observation(
        self,
        image: Image.Image | List[Image.Image] | Dict[str, Image.Image],
        state: np.ndarray | Tensor,
        text: str,
    ) -> Dict[str, Any]:
        """
        Prepare a single observation for inference.

        Args:
            image: Single PIL image, list of images (multi-camera), or
                   dict mapping camera names to images
            state: State vector (numpy or tensor)
            text: Task instruction text

        Returns:
            Dict with preprocessed tensors ready for model
        """
        # Handle different image formats
        if isinstance(image, dict):
            images = {k: [v] for k, v in image.items()}
        elif isinstance(image, list):
            images = image if len(image) == 1 else image[:1]
            images = [images] if not isinstance(images[0], list) else images
            images = images[0]  # Take first camera
        else:
            images = [image]

        # Convert state to tensor
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dim

        # Build batch
        batch = {
            "images": images,
            "states": state,
            "actions": torch.zeros(1, self.config.chunk_size, self.config.max_action_dim),
            "action_is_pad": torch.zeros(1, self.config.chunk_size, dtype=torch.bool),
            "texts": [text],
        }

        return self.preprocessor.preprocess_batch(batch, train=False)

    @torch.no_grad()
    def infer(
        self,
        obs: Dict[str, Any],
        noise: Optional[np.ndarray | Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on an observation dict (OpenPI-style interface).

        Args:
            obs: Dict with keys:
                - "images": Dict[camera_name, np.ndarray [H, W, C]] or single image
                - "state": np.ndarray [state_dim]
                - "prompt" or "task": str

        Returns:
            Dict with:
                - "actions": np.ndarray [T, A]
                - "state": np.ndarray [state_dim]
        """
        self.model.eval()

        # Extract components from obs
        images = obs.get("images", obs.get("image"))
        state = obs.get("state")
        text = obs.get("prompt", obs.get("task", obs.get("text", "")))

        # Convert numpy images to PIL
        if isinstance(images, dict):
            pil_images = {}
            for k, v in images.items():
                if isinstance(v, np.ndarray):
                    if v.max() <= 1.0:
                        v = (v * 255).astype(np.uint8)
                    pil_images[k] = Image.fromarray(v)
                else:
                    pil_images[k] = v
            images = pil_images
        elif isinstance(images, np.ndarray):
            if images.max() <= 1.0:
                images = (images * 255).astype(np.uint8)
            images = Image.fromarray(images)

        # Prepare inputs
        inputs = self._prepare_single_observation(images, state, text)

        # Prepare noise if provided
        if noise is not None:
            if isinstance(noise, np.ndarray):
                noise = torch.from_numpy(noise).to(self.device)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            noise = noise.to(dtype=self.model.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)

        # Run model
        actions = self.model.sample_actions(
            device=torch.device(self.device),
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            noise=noise,
            num_steps=self.config.num_inference_steps,
        )

        # Postprocess
        actions = self.preprocessor.postprocess_actions(actions)
        actions = actions.squeeze(0).cpu().numpy()  # [T, A]

        return {
            "actions": actions,
            "state": state if isinstance(state, np.ndarray) else state.cpu().numpy(),
        }

    @torch.no_grad()
    def select_action(
        self,
        image: Image.Image | List[Image.Image] | Dict[str, Image.Image],
        state: np.ndarray | Tensor,
        text: str,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Convenience method to select action from observation components.

        Args:
            image: Single image, list of images, or dict of images
            state: State vector
            text: Task instruction

        Returns:
            [T, A] action tensor
        """
        obs = {
            "images": image,
            "state": state,
            "prompt": text,
        }
        result = self.infer(obs, noise=noise)
        return torch.from_numpy(result["actions"])

    @torch.no_grad()
    def forward_loss(
        self,
        pixel_values: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute training loss.

        Returns loss with reduction="none" as in OpenPI.
        """
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            actions=actions,
            noise=noise,
            time=time,
        )
