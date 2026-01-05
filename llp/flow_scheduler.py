import torch
import numpy as np

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