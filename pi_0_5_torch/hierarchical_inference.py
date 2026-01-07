"""
Pi0.5 Hierarchical Inference Module

This module provides a high-level API for hierarchical inference with Pi0.5.

The Pi0.5 hierarchical policy operates in two stages:
1. **High-Level Inference**: Given observation + task prompt → predict subtask
   π_θ(ℓ̂ | o_t, ℓ)  e.g., "clean the bedroom" → "pick up pillow"

2. **Low-Level Inference**: Given observation + subtask → predict actions
   π_θ(a_{t:t+H} | o_t, ℓ̂)  e.g., "pick up pillow" → action chunk

Example usage:
    >>> from pi_0_5_torch.hierarchical_inference import Pi05HierarchicalPolicy
    >>> 
    >>> policy = Pi05HierarchicalPolicy.from_pretrained("./checkpoint")
    >>> 
    >>> # Full hierarchical inference
    >>> result = policy.infer(
    ...     image=pil_image,
    ...     state=state_tensor,
    ...     task_prompt="clean the bedroom",
    ... )
    >>> print(f"Subtask: {result.subtask_text}")
    >>> print(f"Actions: {result.actions.shape}")
    >>> 
    >>> # Low-level only (with pre-specified subtask)
    >>> result = policy.infer(
    ...     image=pil_image,
    ...     state=state_tensor,
    ...     subtask="pick up pillow",
    ... )
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from torch import Tensor

from .configuration import PI05Config
from .model import PI05Model, Pi05HierarchicalOutput
from .preprocess import Pi05DatasetStats, Pi05Preprocessor

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalInferenceResult:
    """Result container for hierarchical inference."""
    
    # Predicted actions [chunk_size, action_dim]
    actions: Tensor
    
    # Predicted subtask text (if generated)
    subtask_text: Optional[str] = None
    
    # Raw subtask token ids
    subtask_ids: Optional[Tensor] = None
    
    # Original task prompt
    task_prompt: Optional[str] = None
    
    # Model's raw output
    raw_output: Optional[Pi05HierarchicalOutput] = None


class Pi05HierarchicalPolicy:
    """
    High-level API for Pi0.5 hierarchical policy inference.
    
    This class wraps the Pi0.5 model and provides a simple interface for:
    1. Full hierarchical inference (task → subtask → actions)
    2. Low-level only inference (subtask → actions)
    3. High-level only inference (task → subtask)
    
    Attributes:
        model: The Pi0.5 model
        preprocessor: Data preprocessor
        tokenizer: Tokenizer for text encoding/decoding
        config: Model configuration
    """
    
    def __init__(
        self,
        model: PI05Model,
        preprocessor: Pi05Preprocessor,
        config: PI05Config,
        device: Optional[str] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
        self.tokenizer = preprocessor.tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.model.eval()
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[PI05Config] = None,
        dataset_stats_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "Pi05HierarchicalPolicy":
        """
        Load a hierarchical policy from a checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            config: Optional config (loaded from checkpoint if not provided)
            dataset_stats_path: Optional path to dataset statistics
            device: Device to load model on
            
        Returns:
            Initialized Pi05HierarchicalPolicy
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get config
        if config is None:
            config = checkpoint.get("config", PI05Config())
            if not isinstance(config, PI05Config):
                config = PI05Config(**config) if isinstance(config, dict) else PI05Config()
        
        # Build model
        model = PI05Model(config)
        
        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        
        # Build preprocessor
        stats = Pi05DatasetStats()
        if dataset_stats_path is not None:
            import os
            if os.path.exists(dataset_stats_path):
                stats_data = torch.load(dataset_stats_path, map_location="cpu")
                stats.state_stats = stats_data.get("state")
                stats.action_stats = stats_data.get("action")
        
        preprocessor = Pi05Preprocessor(
            config=config,
            dataset_stats=stats,
            device=device,
            use_augmentation=False,
        )
        
        logger.info(f"Loaded hierarchical policy from {checkpoint_path}")
        
        return cls(model, preprocessor, config, device)
    
    def _prepare_inputs(
        self,
        image: Image.Image,
        state: Tensor,
        prompt: str,
    ) -> dict:
        """Prepare model inputs from raw data."""
        # Create batch format
        batch = {
            "images": [image],
            "states": state.unsqueeze(0) if state.dim() == 1 else state,
            "texts": [prompt],
            "actions": torch.zeros(1, self.config.chunk_size, self.config.max_action_dim),
            "action_is_pad": torch.ones(1, self.config.chunk_size, dtype=torch.bool),
        }
        
        # Preprocess
        inputs = self.preprocessor.preprocess_batch(batch, train=False)
        
        return inputs
    
    @torch.no_grad()
    def generate_subtask(
        self,
        image: Image.Image,
        state: Tensor,
        task_prompt: str,
        max_tokens: int = 32,
        temperature: float = 1.0,
    ) -> tuple[str, Tensor]:
        """
        Generate subtask from task prompt (high-level inference only).
        
        Args:
            image: Input image
            state: Robot state tensor [state_dim]
            task_prompt: High-level task description (e.g., "clean the bedroom")
            max_tokens: Maximum subtask tokens to generate
            temperature: Sampling temperature
            
        Returns:
            (subtask_text, subtask_ids)
        """
        inputs = self._prepare_inputs(image, state, task_prompt)
        
        subtask_ids = self.model.generate_subtask(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Decode to text
        subtask_text = self.tokenizer.decode(subtask_ids[0], skip_special_tokens=True)
        
        return subtask_text, subtask_ids
    
    @torch.no_grad()
    def generate_actions(
        self,
        image: Image.Image,
        state: Tensor,
        subtask: str,
        num_steps: Optional[int] = None,
    ) -> Tensor:
        """
        Generate actions from subtask (low-level inference only).
        
        Args:
            image: Input image
            state: Robot state tensor [state_dim]
            subtask: Subtask description (e.g., "pick up pillow")
            num_steps: Number of flow matching denoising steps
            
        Returns:
            Actions tensor [chunk_size, action_dim]
        """
        inputs = self._prepare_inputs(image, state, subtask)
        
        actions_norm = self.model.sample_actions(
            device=self.device,
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_steps=num_steps,
        )
        
        # Denormalize actions
        actions = self.preprocessor.postprocess_actions(actions_norm)
        
        return actions[0]  # Remove batch dimension
    
    @torch.no_grad()
    def infer(
        self,
        image: Image.Image,
        state: Tensor,
        task_prompt: Optional[str] = None,
        subtask: Optional[str] = None,
        generate_subtask: bool = True,
        max_subtask_tokens: int = 32,
        num_action_steps: Optional[int] = None,
        temperature: float = 1.0,
    ) -> HierarchicalInferenceResult:
        """
        Full hierarchical inference.
        
        If task_prompt is provided and generate_subtask=True:
            1. Generate subtask from task prompt
            2. Generate actions from subtask
            
        If subtask is provided (or generate_subtask=False):
            1. Generate actions from subtask directly
        
        Args:
            image: Input image
            state: Robot state tensor [state_dim]
            task_prompt: High-level task description (optional)
            subtask: Pre-specified subtask (skips high-level inference if provided)
            generate_subtask: Whether to generate subtask from task_prompt
            max_subtask_tokens: Maximum tokens for subtask generation
            num_action_steps: Number of flow matching denoising steps
            temperature: Sampling temperature for subtask generation
            
        Returns:
            HierarchicalInferenceResult with actions and optional subtask
        """
        subtask_text = subtask
        subtask_ids = None
        
        # Step 1: High-level inference (if needed)
        if subtask is None and task_prompt is not None and generate_subtask:
            subtask_text, subtask_ids = self.generate_subtask(
                image=image,
                state=state,
                task_prompt=task_prompt,
                max_tokens=max_subtask_tokens,
                temperature=temperature,
            )
            logger.info(f"Generated subtask: {subtask_text}")
        
        # Step 2: Low-level inference
        prompt_for_actions = subtask_text or task_prompt or ""
        actions = self.generate_actions(
            image=image,
            state=state,
            subtask=prompt_for_actions,
            num_steps=num_action_steps,
        )
        
        return HierarchicalInferenceResult(
            actions=actions,
            subtask_text=subtask_text,
            subtask_ids=subtask_ids,
            task_prompt=task_prompt,
        )
    
    def run_episode(
        self,
        get_observation_fn,
        execute_action_fn,
        task_prompt: str,
        max_steps: int = 100,
        subtask_interval: int = 50,
        action_stride: int = 1,
    ):
        """
        Run a full episode with hierarchical control.
        
        This implements the Pi0.5 inference procedure where:
        - High-level subtask is predicted every `subtask_interval` steps
        - Low-level actions are executed every `action_stride` steps
        
        Args:
            get_observation_fn: Function that returns (image, state)
            execute_action_fn: Function that executes action and returns done flag
            task_prompt: High-level task description
            max_steps: Maximum episode steps
            subtask_interval: Steps between subtask predictions
            action_stride: Steps between action executions (for action chunking)
            
        Returns:
            List of (subtask, actions) pairs executed during episode
        """
        history = []
        current_subtask = None
        action_buffer = None
        action_idx = 0
        
        for step in range(max_steps):
            image, state = get_observation_fn()
            
            # Predict new subtask if needed
            if step % subtask_interval == 0 or current_subtask is None:
                result = self.infer(
                    image=image,
                    state=state,
                    task_prompt=task_prompt,
                    generate_subtask=True,
                )
                current_subtask = result.subtask_text
                action_buffer = result.actions
                action_idx = 0
                history.append((current_subtask, action_buffer))
                logger.info(f"Step {step}: New subtask = {current_subtask}")
            
            # Get action from buffer or regenerate
            if action_idx >= len(action_buffer) or action_idx >= self.config.chunk_size:
                result = self.infer(
                    image=image,
                    state=state,
                    subtask=current_subtask,
                    generate_subtask=False,
                )
                action_buffer = result.actions
                action_idx = 0
            
            # Execute action
            action = action_buffer[action_idx]
            done = execute_action_fn(action)
            action_idx += action_stride
            
            if done:
                logger.info(f"Episode completed at step {step}")
                break
        
        return history


# Convenience function for quick inference
def infer_hierarchical(
    checkpoint_path: str,
    image: Image.Image,
    state: Tensor,
    task_prompt: str,
    device: Optional[str] = None,
) -> HierarchicalInferenceResult:
    """
    Quick hierarchical inference without explicit policy initialization.
    
    Args:
        checkpoint_path: Path to model checkpoint
        image: Input image
        state: Robot state tensor
        task_prompt: High-level task description
        device: Device to use
        
    Returns:
        HierarchicalInferenceResult
    """
    policy = Pi05HierarchicalPolicy.from_pretrained(checkpoint_path, device=device)
    return policy.infer(image=image, state=state, task_prompt=task_prompt)

