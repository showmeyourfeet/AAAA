"""
Independent implementations of robomimic modules.
This module provides ResNet18Conv, ResNet50Conv, SpatialSoftmax, and ConditionalUnet1D
to replace dependencies on robomimic library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torchvision.models.resnet import resnet18, resnet50


class ResNet18Conv(nn.Module):
    """
    ResNet18 backbone for image feature extraction.
    Compatible with robomimic.models.base_nets.ResNet18Conv interface.
    """
    
    def __init__(
        self,
        input_channel: int = 3,
        pretrained: bool = False,
        input_coord_conv: bool = False,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.pretrained = pretrained
        self.input_coord_conv = input_coord_conv
        
        # Load pretrained ResNet18 if requested
        if pretrained:
            resnet = resnet18(pretrained=True)
        else:
            resnet = resnet18(pretrained=False)
        
        # Modify first conv layer if input_channel != 3
        if input_channel != 3:
            # Replace first conv layer
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                input_channel,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
        
        # Modify first conv layer if needed (before creating Sequential)
        if input_coord_conv:
            # CoordConv adds 2 channels (x, y coordinates)
            # Match original robomimic: CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract backbone layers (remove avgpool and fc)
        # Match original robomimic: use Sequential with all children except last 2
        self.nets = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Feature tensor of shape [B, C_out, H_out, W_out]
        """
        # Use nets Sequential (matching original robomimic)
        # CoordConv is already part of resnet.conv1 if input_coord_conv=True
        x = self.nets(x)
        
        return x


class ResNet50Conv(nn.Module):
    """
    ResNet50 backbone for image feature extraction.
    Compatible with robomimic.models.base_nets.ResNet50Conv interface.
    """
    
    def __init__(
        self,
        input_channel: int = 3,
        pretrained: bool = False,
        input_coord_conv: bool = False,
    ):
        super().__init__()
        self.input_channel = input_channel
        self.pretrained = pretrained
        self.input_coord_conv = input_coord_conv
        
        # Load pretrained ResNet50 if requested
        if pretrained:
            resnet = resnet50(pretrained=True)
        else:
            resnet = resnet50(pretrained=False)
        
        # Modify first conv layer if needed (before creating Sequential)
        if input_coord_conv:
            # Match original robomimic: CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract backbone layers (remove avgpool and fc)
        # Match original robomimic: use Sequential with all children except last 2
        self.nets = torch.nn.Sequential(*(list(resnet.children())[:-2]))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Use nets Sequential (matching original robomimic)
        # CoordConv is already part of resnet.conv1 if input_coord_conv=True
        x = self.nets(x)
        
        return x


class CoordConv2d(nn.Conv2d):
    """
    2D Coordinate Convolution - matches original robomimic implementation.
    
    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        assert coord_encoding == 'position', "Only 'position' encoding is supported"
        self.coord_encoding = coord_encoding
        
        # Add 2 channels for positional encoding
        in_channels += 2
        self._position_enc = None  # Will be created on first forward pass
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add coordinate channels and apply convolution.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create position encoding (cached for efficiency)
        if self._position_enc is None:
            pos_y, pos_x = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            pos_y = pos_y.float() / float(H)  # Normalize to [0, 1]
            pos_x = pos_x.float() / float(W)  # Normalize to [0, 1]
            self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)  # [1, 2, H, W]
        
        # Expand position encoding to batch size
        pos_enc = self._position_enc.expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # Concatenate with input
        x_with_coords = torch.cat([x, pos_enc], dim=1)  # [B, C+2, H, W]
        
        return super().forward(x_with_coords)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer that converts feature maps to keypoint coordinates.
    Compatible with robomimic.models.base_nets.SpatialSoftmax interface.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        num_kp: int = 32,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_kp = num_kp
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        self.noise_std = noise_std
        
        C, H, W = input_shape
        
        # Create keypoint prediction layer
        self.conv = nn.Conv2d(C, num_kp, kernel_size=1)
        
        # Learnable temperature parameter
        if learnable_temperature:
            self.temperature_param = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature_param', torch.tensor(temperature))
        
        # Create coordinate grids for spatial softmax
        # Match original robomimic: use np.linspace(-1., 1., ...) for coordinates
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., W),
            np.linspace(-1., 1., H)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, H * W)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, H * W)).float()
        
        # Register as buffer (not a parameter)
        self.register_buffer('pos_x', pos_x)  # [1, H*W]
        self.register_buffer('pos_y', pos_y)  # [1, H*W]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Keypoint coordinates [B, num_kp, 2] where 2 is (x, y)
        """
        B, C, H, W = x.shape
        
        # Predict keypoint heatmaps
        heatmaps = self.conv(x)  # [B, num_kp, H, W]
        
        # Add noise during training if specified
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(heatmaps) * self.noise_std
            heatmaps = heatmaps + noise
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, self.num_kp, -1)  # [B, num_kp, H*W]
        
        # Apply softmax with temperature (matching original robomimic)
        temperature = self.temperature_param if self.learnable_temperature else self.temperature
        attention = F.softmax(heatmaps_flat / temperature, dim=-1)  # [B, num_kp, H*W]
        
        # Compute expected x and y coordinates using attention weights
        # [1, H*W] x [B*num_kp, H*W] -> [B*num_kp, 1]
        expected_x = torch.sum(self.pos_x * attention.view(-1, H * W), dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention.view(-1, H * W), dim=1, keepdim=True)
        
        # Stack to [B*num_kp, 2] and reshape to [B, num_kp, 2]
        expected_xy = torch.cat([expected_x, expected_y], dim=1)
        keypoints = expected_xy.view(B, self.num_kp, 2)
        
        return keypoints


class ConditionalUnet1D(nn.Module):
    """
    1D Conditional U-Net for diffusion policy.
    Compatible with robomimic.models.diffusion_policy_nets.ConditionalUnet1D interface.
    
    This is a simplified 1D U-Net architecture for predicting noise in diffusion models.
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: Optional[int] = None,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        diffusion_step_embed_dim: int = 128,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        self.down_dims = down_dims
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        
        # Global condition projection
        if global_cond_dim is not None and global_cond_dim > 0:
            self.global_cond_proj = nn.Linear(global_cond_dim, diffusion_step_embed_dim)
        else:
            self.global_cond_proj = None
        
        # Input projection
        in_dim = input_dim
        if global_cond_dim is not None and global_cond_dim > 0:
            in_dim += diffusion_step_embed_dim
        
        # Helper function to get safe GroupNorm num_groups
        def get_num_groups(channels, default=8):
            """Get num_groups that divides channels."""
            num_groups = min(default, channels)
            while channels % num_groups != 0:
                num_groups -= 1
            return max(1, num_groups)
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        prev_dim = in_dim
        for dim in down_dims:
            num_groups = get_num_groups(dim)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(prev_dim, dim, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(num_groups, dim),
                    nn.SiLU(),
                    nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(num_groups, dim),
                    nn.SiLU(),
                )
            )
            prev_dim = dim
        
        # Bottleneck
        num_groups_bottleneck = get_num_groups(prev_dim)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(prev_dim, prev_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups_bottleneck, prev_dim),
            nn.SiLU(),
            nn.Conv1d(prev_dim, prev_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups_bottleneck, prev_dim),
            nn.SiLU(),
        )
        
        # Decoder (upsampling)
        # Decoder should mirror encoder: start from bottleneck and go back through down_dims in reverse
        self.decoder = nn.ModuleList()
        # First decoder layer: bottleneck -> last encoder dim
        # Then go through down_dims in reverse order
        decoder_dims = list(reversed(down_dims))
        for i, decoder_dim in enumerate(decoder_dims):
            # Input to decoder layer: previous decoder output + corresponding encoder output
            # First layer: bottleneck (prev_dim) + last encoder output (decoder_dim)
            # Subsequent layers: previous decoder output + corresponding encoder output
            num_groups = get_num_groups(decoder_dim)
            self.decoder.append(
                nn.Sequential(
                    nn.Conv1d(prev_dim + decoder_dim, decoder_dim, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(num_groups, decoder_dim),
                    nn.SiLU(),
                    nn.Conv1d(decoder_dim, decoder_dim, kernel_size, padding=kernel_size // 2),
                    nn.GroupNorm(num_groups, decoder_dim),
                    nn.SiLU(),
                )
            )
            prev_dim = decoder_dim
        
        # Output projection
        self.output_proj = nn.Conv1d(prev_dim, input_dim, kernel_size=1)
        
    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sample: Noisy action [B, T, input_dim] or [B, input_dim, T]
            timestep: Diffusion timestep [B] or scalar
            global_cond: Global condition [B, global_cond_dim]
            
        Returns:
            Predicted noise [B, T, input_dim] or [B, input_dim, T]
        """
        # Handle input shape: support both [B, T, D] and [B, D, T]
        # Check if input is [B, T, D] format (last dim matches input_dim)
        if sample.dim() == 3:
            if sample.shape[-1] == self.input_dim:
                # [B, T, D] -> [B, D, T]
                sample = sample.transpose(1, 2)
            elif sample.shape[1] == self.input_dim:
                # Already [B, D, T] format
                pass
            else:
                raise ValueError(
                    f"Input shape mismatch: expected [B, T, {self.input_dim}] or [B, {self.input_dim}, T], "
                    f"got {sample.shape}"
                )
        elif sample.dim() == 2:
            # [B, D] -> [B, D, 1] (single timestep)
            if sample.shape[1] == self.input_dim:
                sample = sample.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Input shape mismatch: expected [B, {self.input_dim}], got {sample.shape}"
                )
        else:
            raise ValueError(f"Input must be 2D or 3D tensor, got {sample.dim()}D")
        
        B, D, T = sample.shape
        
        # Time embedding
        # Handle timestep: can be scalar, 1D tensor, or multi-element tensor
        if isinstance(timestep, (int, float)):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        elif isinstance(timestep, torch.Tensor):
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0)
            # Ensure timestep is on the same device
            timestep = timestep.to(sample.device)
            # If timestep has multiple elements but batch size is 1, use first element
            if timestep.shape[0] != B:
                if timestep.numel() == 1:
                    timestep = timestep.expand(B)
                else:
                    # Use first timestep for all samples (common in inference)
                    timestep = timestep[0:1].expand(B)
        
        # Sinusoidal time embedding
        # Ensure half_dim is at least 1 to avoid division by zero
        half_dim = max(1, self.diffusion_step_embed_dim // 2)
        emb = np.log(10000.0) / max(1, half_dim - 1)  # Avoid division by zero
        emb = torch.exp(torch.arange(half_dim, device=sample.device, dtype=sample.dtype) * -emb)
        emb = timestep[:, None].float() * emb[None, :]
        time_emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, 2*half_dim]
        
        # Pad if needed to match diffusion_step_embed_dim (handles odd dimensions)
        if time_emb.shape[-1] < self.diffusion_step_embed_dim:
            padding = torch.zeros(
                time_emb.shape[0], 
                self.diffusion_step_embed_dim - time_emb.shape[-1],
                device=time_emb.device,
                dtype=time_emb.dtype
            )
            time_emb = torch.cat([time_emb, padding], dim=-1)
        
        time_emb = self.time_embed(time_emb)  # [B, diffusion_step_embed_dim]
        
        # Global condition
        if self.global_cond_proj is not None and global_cond is not None:
            cond_emb = self.global_cond_proj(global_cond)  # [B, diffusion_step_embed_dim]
            # Combine time and condition embeddings
            combined_emb = time_emb + cond_emb  # [B, diffusion_step_embed_dim]
        else:
            combined_emb = time_emb
        
        # Expand condition to match spatial dimensions
        combined_emb = combined_emb.unsqueeze(-1).expand(-1, -1, T)  # [B, diffusion_step_embed_dim, T]
        
        # Concatenate with input
        x = torch.cat([sample, combined_emb], dim=1)  # [B, input_dim + diffusion_step_embed_dim, T]
        
        # Encoder
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            # Concatenate with corresponding encoder output
            skip = encoder_outputs[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        # Output projection
        output = self.output_proj(x)  # [B, input_dim, T]
        
        # Convert back to [B, T, input_dim] if needed
        # Note: robomimic's ConditionalUnet1D typically returns [B, T, D]
        output = output.transpose(1, 2)  # [B, T, input_dim]
        
        return output


def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """
    Replace BatchNorm layers with GroupNorm.
    Compatible with robomimic.algo.diffusion_policy.replace_bn_with_gn interface.
    
    Args:
        module: PyTorch module to process
        num_groups: Number of groups for GroupNorm
        
    Returns:
        Modified module (in-place modification)
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            # Ensure num_groups divides num_channels
            actual_groups = min(num_groups, num_channels)
            while num_channels % actual_groups != 0:
                actual_groups -= 1
            if actual_groups < 1:
                actual_groups = 1
            
            gn = nn.GroupNorm(actual_groups, num_channels, eps=child.eps, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)
    return module

