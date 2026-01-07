"""
Independent implementations of diffusers modules.
This module provides DDIMScheduler and EMAModel to replace dependencies on diffusers library.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class SchedulerOutput:
    """Output of scheduler step."""
    prev_sample: torch.Tensor


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) Scheduler.
    Compatible with diffusers.schedulers.scheduling_ddim.DDIMScheduler interface.
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.set_alpha_to_one = set_alpha_to_one
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        
        # Initialize betas
        self.betas = self._get_betas()
        
        # Compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        
        # For set_alpha_to_one: store final_alpha_cumprod separately (like original diffusers)
        # This is used when prev_timestep < 0
        self.final_alpha_cumprod = np.array([1.0]) if set_alpha_to_one else self.alphas_cumprod[0:1]
        
        # Standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0
        
        # Inference timesteps (set by set_timesteps)
        self.timesteps = None
        self.num_inference_steps = None
        
        # Store config for compatibility
        self.config = {
            "num_train_timesteps": num_train_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            "clip_sample": clip_sample,
            "set_alpha_to_one": set_alpha_to_one,
            "steps_offset": steps_offset,
            "prediction_type": prediction_type,
        }
    
    def _get_betas(self) -> np.ndarray:
        """Compute betas based on schedule."""
        if self.beta_schedule == "linear":
            return np.linspace(self.beta_start, self.beta_end, self.num_train_timesteps, dtype=np.float32)
        elif self.beta_schedule == "scaled_linear":
            # Scale linear schedule
            return np.linspace(
                self.beta_start**0.5, self.beta_end**0.5, self.num_train_timesteps, dtype=np.float32
            ) ** 2
        elif self.beta_schedule == "squaredcos_cap_v2":
            # Cosine schedule with squared cap
            return self._cosine_beta_schedule()
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")
    
    def _cosine_beta_schedule(self) -> np.ndarray:
        """Cosine beta schedule - matches diffusers betas_for_alpha_bar function."""
        def alpha_bar(time_step):
            return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2
        
        betas = []
        for i in range(self.num_train_timesteps):
            t1 = i / self.num_train_timesteps
            t2 = (i + 1) / self.num_train_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float32)
    
    def set_timesteps(self, num_inference_steps: int, device: Optional[Union[torch.device, str]] = None):
        """
        Set the discrete timesteps used for the diffusion chain.
        
        Args:
            num_inference_steps: Number of diffusion steps used when generating samples
            device: Device to place timesteps on (can be string like 'cuda' or torch.device)
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        
        # Create timesteps: evenly spaced from num_train_timesteps to 0
        # Match original diffusers: create integer timesteps by multiplying by ratio
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        
        # Convert to tensor and add steps_offset (matching original diffusers order)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset
        
        # Ensure timesteps are within valid range
        self.timesteps = torch.clamp(self.timesteps, 0, self.num_train_timesteps - 1)
    
    def scale_model_input(self, sample: torch.Tensor, timestep: int) -> torch.Tensor:
        """
        Scale model input according to the current timestep.
        For DDIM, this is a no-op.
        """
        return sample
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples.
        
        Args:
            original_samples: Original samples [B, ...]
            noise: Noise to add [B, ...]
            timesteps: Timesteps [B] or scalar
            
        Returns:
            Noisy samples [B, ...]
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Convert alphas_cumprod to tensor if needed
        if isinstance(self.alphas_cumprod, np.ndarray):
            self.alphas_cumprod = torch.from_numpy(self.alphas_cumprod).to(
                device=original_samples.device, dtype=original_samples.dtype
            )
        else:
            self.alphas_cumprod = self.alphas_cumprod.to(
                device=original_samples.device, dtype=original_samples.dtype
            )
        
        timesteps = timesteps.to(original_samples.device)
        
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
    ) -> SchedulerOutput:
        """
        Predict the sample at the previous timestep.
        
        Args:
            model_output: Predicted noise/residual [B, ...]
            timestep: Current timestep (scalar or tensor)
            sample: Current sample [B, ...]
            eta: DDIM eta parameter (0 = deterministic DDIM, 1 = DDPM)
            use_clipped_model_output: Whether to clip model output
            
        Returns:
            SchedulerOutput with prev_sample
        """
        # Convert timestep to int
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                timestep = timestep.item()
            else:
                # If multiple timesteps, use the first one
                timestep = timestep[0].item()
        
        # Get previous timestep - match original diffusers implementation
        # Original: prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        
        # Get alpha_cumprod values (matching original diffusers)
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod[0]
        
        # Convert to tensors
        device = sample.device
        alpha_prod_t = torch.tensor(alpha_prod_t, device=device, dtype=sample.dtype)
        alpha_prod_t_prev = torch.tensor(alpha_prod_t_prev, device=device, dtype=sample.dtype)
        
        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (1 - alpha_prod_t) ** 0.5 * model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # Clip predicted sample if needed
        if self.clip_sample or use_clipped_model_output:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute variance (DDIM formula) - matching original diffusers _get_variance
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        # Clamp variance to ensure it's non-negative
        variance = max(0.0, variance)
        
        std_dev_t = eta * variance ** 0.5
        
        # Compute pred_sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        
        # Add noise for stochastic sampling (if eta > 0)
        if eta > 0:
            noise = torch.randn_like(sample)
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + std_dev_t * noise
        else:
            # Deterministic DDIM
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        return SchedulerOutput(prev_sample=prev_sample)


class EMAModel:
    """
    Exponential Moving Average (EMA) model wrapper.
    Compatible with diffusers.training_utils.EMAModel interface.
    """
    
    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
        max_value: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EMA model.
        
        Args:
            model: Model to wrap
            update_after_step: Number of steps to wait before starting EMA updates
            inv_gamma: Inverse multiplicative factor of EMA warmup
            power: Exponential factor of EMA warmup (default 2/3)
            min_value: Minimum EMA decay rate
            max_value: Maximum EMA decay rate
            device: Device to move averaged model to
        """
        import copy
        
        # Create averaged model by copying the original model
        # Match original diffusers: use eval() and requires_grad_(False)
        self.averaged_model = copy.deepcopy(model).eval()
        self.averaged_model.requires_grad_(False)
        
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        
        if device is not None:
            self.averaged_model = self.averaged_model.to(device=device)
        
        # Initialize EMA state
        self.decay = 0.0
        self.optimization_step = 0
    
    def get_decay(self, optimization_step: int) -> float:
        """
        Compute the decay factor for the exponential moving average.
        Matches original diffusers implementation.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        
        if step <= 0:
            return 0.0
        
        return max(self.min_value, min(value, self.max_value))
    
    @torch.no_grad()
    def step(self, new_model: nn.Module):
        """
        Update EMA model with new model weights.
        Matches original diffusers implementation.
        
        Args:
            new_model: Updated model (typically the same as self.model)
        """
        import copy
        
        # Ensure averaged_model is on the same device as new_model
        # Prefer CUDA if available to avoid CPU synchronization issues
        if next(new_model.parameters(), None) is not None:
            model_device = next(new_model.parameters()).device
            
            # Determine target device: prefer CUDA if available
            if torch.cuda.is_available():
                # If model is on CUDA, use that specific CUDA device
                if model_device.type == 'cuda':
                    target_device = model_device
                else:
                    # Model is on CPU but CUDA available - prefer CUDA:0
                    # This ensures EMA operations happen on CUDA for better performance
                    target_device = torch.device('cuda:0')
            else:
                # No CUDA available, use model's device
                target_device = model_device
            
            # Move averaged_model to target device if needed
            if next(self.averaged_model.parameters(), None) is not None:
                ema_device = next(self.averaged_model.parameters()).device
                if ema_device != target_device:
                    self.averaged_model = self.averaged_model.to(device=target_device)
            else:
                # If averaged_model has no parameters, still move it to target device
                self.averaged_model = self.averaged_model.to(device=target_device)
        
        ema_state_dict = {}
        ema_params = self.averaged_model.state_dict()
        
        self.decay = self.get_decay(self.optimization_step)
        
        # Update parameters
        for key, param in new_model.named_parameters():
            if isinstance(param, dict):
                continue
            try:
                ema_param = ema_params[key]
            except KeyError:
                # Handle missing parameters
                ema_param = param.float().clone() if param.ndim == 1 else copy.deepcopy(param)
                ema_params[key] = ema_param
            
            if not param.requires_grad:
                # For non-trainable parameters, just copy the value
                ema_params[key].copy_(param.to(device=ema_param.device, dtype=ema_param.dtype).data)
                ema_param = ema_params[key]
            else:
                # For trainable parameters, apply EMA update
                ema_param.mul_(self.decay)
                ema_param.add_(param.data.to(device=ema_param.device, dtype=ema_param.dtype), alpha=1 - self.decay)
            
            ema_state_dict[key] = ema_param
        
        # Update buffers (buffers are copied directly, not EMA'd)
        for key, param in new_model.named_buffers():
            # Ensure buffer is on the same device as the averaged model
            if key in ema_params:
                ema_buffer = ema_params[key]
                ema_state_dict[key] = param.to(device=ema_buffer.device, dtype=ema_buffer.dtype)
            else:
                ema_state_dict[key] = param
        
        # Load updated state dict
        self.averaged_model.load_state_dict(ema_state_dict, strict=False)
        self.optimization_step += 1
    
    def copy_to(self, model: nn.Module):
        """
        Copy averaged model weights to target model.
        
        Args:
            model: Target model to copy weights to
        """
        model.load_state_dict(self.averaged_model.state_dict())
    
    def to(self, device: torch.device):
        """Move EMA model to device."""
        self.averaged_model = self.averaged_model.to(device)
        return self


class FlowMatchEulerDiscreteScheduler:
    """
    Euler Flow Matching Discrete Scheduler.
    Based on the Rectified Flow Theory: dx/dt = v(x, t)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0, # generally equals to 3.0 in SD, but in the simple robot action space, 1.0 is enough.
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(
        self, 
        num_inference_timesteps: int,
        device: str = "cuda",
    ):
        self.num_inference_timesteps = num_inference_timesteps

        # generate timesteps: from 1.0 (pure noise) to 0.0 (pure signal)
        timesteps = np.linspace(1.0, 0.0, num_inference_timesteps + 1) # [1.0, 0.9, ..., 0.0]

        # apply shift (alternatively)
        # simple linear shift Time scaling: t_new = t / (1 + (shift - 1) * t)
        sigmas = 1.0 / (1.0 + (self.shift - 1.0) * timesteps)

        # convert to tensor 
        self.sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        self.timesteps = self.sigmas * self.num_train_timesteps # scaled to [0, 1000] for the input of the diffusion model

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict:bool = False,
        step_index: int = None,
    ):
        """
        Process one step of the Euler Integration: x_{t-1} = x_t + (t_{next} - t_{curr}) * v_model
        Args:
            model_output: the predicted Vector Field v(x_t, t)
            timestep: the current timestep
            sample: the current sample x_t
            return_dict: whether to return a dictionary or a tuple
            step_index: optional index in timesteps array (for efficiency, avoids search)
        """
        # Use provided index if available, otherwise search for it
        if step_index is not None:
            if step_index >= len(self.sigmas) - 1:
                # Already at the end, return sample as-is
                return (sample,)
        else:
            # Find the index of the current timestep in the sigmas tensor
            # Ensure timestep is a scalar tensor for comparison
            if isinstance(timestep, torch.Tensor):
                if timestep.numel() > 1:
                    timestep = timestep[0]  # take first element if tensor has multiple values
            else:
                timestep = torch.tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
            
            step_index = (self.timesteps == timestep).nonzero(as_tuple=False)
            if step_index.numel() > 0:
                step_index = step_index[0].item()
            else:
                # if cannot find the totally matched float timestep, find the nearest timestep
                step_index = (torch.abs(self.timesteps - timestep)).argmin().item()

        sigma_curr = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1] if step_index < len(self.sigmas) - 1 else 0.0

        # Euler Step
        # dt = sigma_next - sigma_curr (generally negative, as we traverse from 1 to 0)
        dt = sigma_next - sigma_curr

        prev_sample = sample + model_output * dt

        # convert prev_sample to the same dtype as model_output
        prev_sample = prev_sample.to(model_output.dtype)

        return (prev_sample, )

    def add_noise(
        self,
        original_sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """
        Noise adding function for the forward process (Rectified Flow Interpolation).
        x_t = t * x_1 (noise) + (1-t) * x_0 (data)  defined by some papers,
        or defined by the diffusers: x_t = sigma * x_1 (noise) + (1-sigma) * x_0 (data).
        Training of Flow Matching is very simple, just linearly interpolate between the noise and the data.
        Note: timestep is [0, 1000], which should be scaled to [0, 1] for normalization.
        """
        # Scale timestep to [0, 1]
        sigmas = timesteps / self.num_train_timesteps

        # Expand dimensions for broadcasting
        while len(sigmas.shape) < len(original_sample.shape):
            sigmas = sigmas.unsqueeze(-1)
        
        # The definition of the Rectified Flow is generally:
        # t=0 -> data, t=1 -> noise (or reverse. In the diffusers, sigma=1 -> noise)
        # Here follows the diffusers' definition: sigma=1 -> noise, sigma=0 -> data.
        # x_t = sigma * noise + (1 - sigma) * data
        noisy_samples = sigmas * noise + (1.0 - sigmas) * original_sample
        return noisy_samples

