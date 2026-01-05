"""Policy definitions for ACT, Diffusion, and Flow Matching."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
from torch.distributions import Beta

from .model import build_act_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args_override: dict) -> None:
        super().__init__()
        model, optimizer = build_act_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override.get("kl_weight", 0.0) or 0.0
        self.vq = args_override.get("vq", False)
        print(f"KL Weight {self.kl_weight}")
        multi_gpu = args_override.get("multi_gpu", False)
        self.num_queries = self.model.module.num_queries if multi_gpu else self.model.num_queries

    def __call__(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
        command_embedding: torch.Tensor | None = None,
        vq_sample: torch.Tensor | None = None,
        encode_command: bool = False,
    ):
        """
        Forward pass for training or inference.
        
        Args:
            encode_command: whether to encode command_embedding in encoder (default False).
                           Set to True to enable command encoding in CVAE encoder.
                           Note: This only affects the encoder; decoder will still use command_embedding
                           if provided and use_language=True.
        """
        env_state = None
        if actions is not None:  # training time
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(
                qpos,
                image,
                env_state,
                actions,
                is_pad,
                command_embedding=command_embedding,
                vq_sample=vq_sample,
                encode_command=encode_command,
            )
            loss_dict: dict[str, torch.Tensor] = {}
            
            # KL divergence: only computed when encoder is available and not using VQ
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0, device=a_hat.device)]
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            
            # VQ discrepancy loss: encourages probs to match binaries
            if self.vq:
                loss_dict["vq_discrepancy"] = F.l1_loss(probs, binaries, reduction="mean")
            
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")

            # l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            mask = (~is_pad).unsqueeze(-1) # [batch, seq, 1]
            # valid should count all valid action dimensions, not just time steps
            # mask.sum() gives number of valid time steps, but each time step has action_dim elements
            action_dim = actions.shape[-1]
            valid = mask.sum() * action_dim  # total number of valid (non-padded) action elements
            l1 = (all_l1 * mask).sum() / valid.clamp(min=1) 

            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _), _, _ = self.model(
                qpos, image, env_state, vq_sample=vq_sample, command_embedding=command_embedding,
                encode_command=encode_command,
            )  # no action, sample from prior
            return a_hat

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        """
        Encode actions to VQ binaries (discrete codes) for inference.
        
        Args:
            qpos: batch, qpos_dim
            actions: batch, seq, action_dim
            is_pad: batch, seq
        
        Returns:
            binaries: batch, vq_class, vq_dim (discrete VQ codes)
        """
        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries]
        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)
        return binaries

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict, strict=True):
        """
        Load model state dict.
        
        Args:
            model_dict: State dict to load
            strict: If False, allows missing/unexpected keys (useful for no_encoder mode)
        
        Returns:
            NamedTuple with missing_keys and unexpected_keys
        """
        # Handle pos_table size mismatch due to different use_state/num_queries config
        # pos_table is deterministically computed from config, so we can safely skip loading
        # if sizes don't match (the model will use its own freshly computed pos_table)
        pos_table_key = "model.pos_table"
        if pos_table_key in model_dict:
            ckpt_shape = model_dict[pos_table_key].shape
            model_shape = self.model.pos_table.shape
            if ckpt_shape != model_shape:
                print(
                    f"Warning: pos_table shape mismatch - checkpoint {ckpt_shape} vs "
                    f"model {model_shape}. This usually means use_state or num_queries "
                    f"(chunk_size) differs between training and inference. "
                    f"Using model's computed pos_table instead."
                )
                del model_dict[pos_table_key]
        return self.load_state_dict(model_dict, strict=strict)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# ============================================================================
# Diffusion Policy
# ============================================================================

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override: dict) -> None:
        super().__init__()
        from robomimic.models.base_nets import ResNet18Conv, ResNet50Conv, SpatialSoftmax
        from robomimic.models.diffusion_policy_nets import ConditionalUnet1D
        from robomimic.algo.diffusion_policy import replace_bn_with_gn
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusers.training_utils import EMAModel
        from llp.backbone_impl import FilMedBackbone


        self.camera_names = args_override['camera_names']
        self.observation_horizon = args_override.get('observation_horizon', 1)
        self.action_horizon = args_override.get("action_horizon", args_override.get("num_queries", 30))      #  close to action chunk size
        self.prediction_horizon = args_override.get("prediction_horizon", self.action_horizon)
        
        # num_queries 用于兼容 train.py 的接口
        self.num_queries = self.action_horizon
        
        # 警告：若 observation_horizon > 1，需要确保 __call__ 中输入维度正确展平
        if self.observation_horizon > 1:
             print(f"Warning: observation_horizon is {self.observation_horizon}. Ensure input formatting flattens history correctly.")

        self.num_inference_timesteps = args_override.get("num_inference_timesteps", 16)
        self.ema_power = args_override.get("ema_power", 0.75)
        self.lr = args_override["lr"]
        self.weight_decay = args_override.get("weight_decay", 1e-4)

        backbone_name = args_override["backbone"]
        use_language = args_override.get("use_language", False)

        self.num_kp = 32
        self.feature_dim = 64
        self.action_dim = args_override.get("action_dim", 20)

        # Get image size from config
        # Priority: (image_height, image_width) > image_size > default 224
        # This allows non-square images (e.g., 640x360) to be properly handled
        img_h = args_override.get("image_height", None)
        img_w = args_override.get("image_width", None)
        
        if img_h is not None and img_w is not None:
            # Use explicit height/width if provided
            pass
        elif "image_size" in args_override:
            # Fallback to image_size if height/width not specified
            image_size = args_override["image_size"]
            if isinstance(image_size, int):
                img_h = img_w = image_size
            elif isinstance(image_size, (tuple, list)):
                img_h, img_w = image_size
            else:
                img_h = img_w = 224  # Default fallback
        else:
            # Final fallback: default to 224x224
            img_h = img_w = 224

        dummy_input = torch.randn(1, 3, img_h, img_w)
        if use_language:
            temp_backbone = FilMedBackbone(backbone_name)
            dummy_cmd = torch.randn(1, 768)
            with torch.no_grad():
                dummy_feature = temp_backbone(dummy_input, dummy_cmd)
        else:
            temp_backbone = ResNet18Conv(**{
                "input_channel": 3,
                "pretrained": False,
                "input_coord_conv": False,
            })
            with torch.no_grad():
                dummy_feature = temp_backbone(dummy_input)

        _, c_out, h_out, w_out = dummy_feature.shape
        input_shape = (c_out, h_out, w_out)
        print(f"Policy Type: Diffusion")
        print(f"Using FiLMed Backbone: {use_language}")
        if use_language:
            print(f"Backbone type: {backbone_name}")
        print(f"Input size: {img_h}x{img_w}")
        print(f"Output shape: {input_shape}")

        del temp_backbone

        backbones = []
        pools = []
        linears = []

        for _ in self.camera_names:
            if use_language:
                # initialize the FiLMed backbone
                backbone = FilMedBackbone(backbone_name)
            else:
                backbone = ResNet18Conv(**{
                    "input_channel": 3,
                    "pretrained": False,
                    "input_coord_conv": False,
                })
            
            backbones.append(backbone)

            # compress the feature map to keypoints
            pools.append(
                SpatialSoftmax(**{
                    "input_shape": input_shape,
                    "num_kp": self.num_kp,
                    "temperature": 1.0,
                    "learnable_temperature": False,
                    "noise_std": 0.0, # Modified: False -> 0.0 for safety
                })
            )

            # project to feature_dim
            linears.append(
                nn.Linear(
                    int(np.prod([self.num_kp, 2])), 
                    self.feature_dim
                )
            )

        self.backbones = nn.ModuleList(backbones)
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)

        # replace the batch norm with group norm to fit the small batch size
        # Use a safer replacement that handles channels not divisible by 16
        self.backbones = self._replace_bn_with_gn_safe(self.backbones)

        # Get use_state flag
        self.use_state = args_override.get("use_state", True)

        # calculate Observation Dimension
        # Only include state_dim if use_state is True
        state_dim = args_override.get("state_dim", 20)
        self.observation_dim = self.feature_dim * len(self.camera_names) + (state_dim if self.use_state else 0)
        
        # 定义语言嵌入维度
        self.lang_embed_dim = 768
        
        # If using language, add projected language embedding dimension to observation_dim
        # The language embedding is projected to feature_dim before concatenation
        if use_language:
            self.observation_dim += self.feature_dim  # Add projected language embedding dimension

        # core diffusion model: Conditional U-Net
        # Note: observation_condition will be flattened if observation_horizon > 1
        self.noise_pred_net = ConditionalUnet1D(
            input_dim = self.action_dim,
            global_cond_dim = self.observation_dim * self.observation_horizon,
        )

        nets_dict = {
            "backbones": self.backbones,
            "pools": self.pools,
            "linears": self.linears,
            "noise_pred_net": self.noise_pred_net,
        }

        if use_language:
            # Modified: 定义层必须在赋值之前
            self.lang_embed_proj = nn.Linear(
                self.lang_embed_dim, self.feature_dim
            )
            nets_dict["lang_embed_proj"] = self.lang_embed_proj

        self.nets = nn.ModuleDict({"policy": nn.ModuleDict(nets_dict)})

        # Note: deleted DataParallel, it breaks the dictionary structure
        # if multi-gpu training, we should handle it externally
        # if args_override.get("multi_gpu", False):
        #     nets = nn.DataParallel(nets)
        
        device = args_override.get("device", "cuda")
        self.nets.to(device)
        # Note: nn.ModuleDict defaults to float32, so no need to explicitly convert

        # add ema model
        self.ema = EMAModel(
            model = self.nets,
            power = self.ema_power
        )

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps = 50,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def _replace_bn_with_gn_safe(self, module_list):
        """
        Safely replace BatchNorm with GroupNorm, handling cases where
        num_channels is not divisible by the desired num_groups.
        """
        def safe_replace_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    num_channels = child.num_features
                    # Try to find a suitable num_groups
                    # Prefer groups of 8, 16, or 32, but ensure divisibility
                    num_groups = None
                    for preferred_groups in [32, 16, 8]:
                        if num_channels % preferred_groups == 0:
                            num_groups = preferred_groups
                            break
                    
                    # If no preferred group size works, use the largest divisor <= 32
                    if num_groups is None:
                        for g in range(32, 0, -1):
                            if num_channels % g == 0:
                                num_groups = g
                                break
                    
                    # Fallback: use num_channels (equivalent to LayerNorm)
                    if num_groups is None:
                        num_groups = num_channels
                    
                    # Create GroupNorm with matching affine parameters
                    gn = nn.GroupNorm(num_groups, num_channels, eps=child.eps, affine=child.affine)
                    if child.affine:
                        gn.weight.data.copy_(child.weight.data)
                        gn.bias.data.copy_(child.bias.data)
                    setattr(module, name, gn)
                else:
                    # Recursively process child modules
                    safe_replace_bn(child)
            return module
        
        for i, backbone in enumerate(module_list):
            safe_replace_bn(backbone)
        return module_list

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.nets.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    
    def __call__(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        command_embedding=None,
        encode_command=False,  # Added for interface compatibility (not used in diffusion policy)
        **kwargs
    ):
        """
        Forward pass is compatible with ACT interface.
        Diffusion policy do not need is_pad to mask Loss. Here is just for interface compatibility.
        
        Args:
            encode_command: Ignored for diffusion policy (kept for interface compatibility)
        """

        B = qpos.shape[0]

        nets = self.nets
        all_features = []

        for cam_id, _ in enumerate(self.camera_names):
            cam_image = image[:, cam_id]

            if "lang_embed_proj" in nets["policy"]:
                cam_features = nets["policy"]["backbones"][cam_id](cam_image, command_embedding)
            else:
                cam_features = nets["policy"]["backbones"][cam_id](cam_image)

            pool_features = nets["policy"]["pools"][cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = nets["policy"]["linears"][cam_id](pool_features)
            all_features.append(out_features)

        # Concatenate features
        # Only concatenate qpos if use_state is True
        features_to_concat = all_features.copy()
        if self.use_state:
            features_to_concat.append(qpos)
        
        if command_embedding is not None and "lang_embed_proj" in nets["policy"]:
            command_embedding_proj = nets["policy"]["lang_embed_proj"](command_embedding)
            features_to_concat.append(command_embedding_proj)
        
        observation_condition = torch.cat(features_to_concat, dim=1)
        
        # Debug: Verify observation_condition dimension matches expected
        actual_obs_dim = observation_condition.shape[1]
        expected_obs_dim = self.observation_dim
        if actual_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"Observation dimension mismatch! "
                f"Actual: {actual_obs_dim}, Expected: {expected_obs_dim}. "
                f"This suggests a bug in observation_dim calculation. "
                f"all_features dims: {[f.shape[1] for f in all_features]}, "
                f"qpos dim: {qpos.shape[1] if self.use_state else 'N/A (use_state=False)'}, "
                f"use_state: {self.use_state}, "
                f"use_language: {command_embedding is not None and 'lang_embed_proj' in nets['policy']}"
            )
        
        # Handle observation_horizon > 1: repeat observation_condition for history
        # Note: This is a simple implementation. For proper history handling, 
        # you may need to modify the data loading to provide historical observations.
        if self.observation_horizon > 1:
            # Repeat the current observation to match observation_horizon
            # In a proper implementation, you would concatenate historical observations here
            observation_condition = observation_condition.unsqueeze(1).repeat(1, self.observation_horizon, 1)
            observation_condition = observation_condition.reshape(B, -1)  # Flatten: [B, observation_dim * observation_horizon]

        # ------------------------------------------- Training Mode -------------------------------------------
        if actions is not None:
            # sample noise
            noise = torch.randn(actions.shape, device=observation_condition.device)

            # sample diffusion timesteps
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (B,),
                device=observation_condition.device,
            ).long()


            # add noise to actions (Forward Diffusion Process)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            # predict noise (Reverse Diffusion Step prediction)
            noise_prediction = nets["policy"]["noise_pred_net"](
                noisy_actions,
                timesteps,
                global_cond=observation_condition
            )

            # calculate loss (MSE)
            all_l2 = F.mse_loss(
                noise_prediction, 
                noise, 
                reduction="none"
            )

            if is_pad is not None:
                # loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()  
                mask = (~is_pad).unsqueeze(-1) # [batch, seq, 1]
                # valid should count all valid action dimensions, not just time steps
                # mask.sum() gives number of valid time steps, but each time step has action_dim elements
                action_dim = actions.shape[-1]
                valid = mask.sum() * action_dim  # total number of valid (non-padded) action elements
                loss = (all_l2 * mask).sum() / valid.clamp(min=1)
            else:
                loss = all_l2.mean()

            # Update EMA weights during training
            if self.training and self.ema is not None:
                # EMAModel.step expects the updated model, not parameters
                self.ema.step(self.nets)

            return {
                "loss": loss,
            }
        else:
            # ------------------------------------------- Inference Mode -------------------------------------------
            # init action (Gaussian Prior)
            # Use EMA-averaged weights for inference if available
            nets = self.ema.averaged_model if self.ema is not None else self.nets
            Tp = self.prediction_horizon
            action_dim = self.action_dim

            noisy_actions = torch.randn(
                (B, Tp, action_dim),
                device = observation_condition.device,
            )
            naction = noisy_actions

            # configure diffusion steps (less than training steps, accelerate inference)
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

            # reverse diffusion process step by step
            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = nets["policy"]["noise_pred_net"](
                    sample=naction, timestep=k, global_cond=observation_condition
                )

                # reverse diffusion (remove noise)
                naction = self.noise_scheduler.step(
                    noise_pred, 
                    k, 
                    naction
                ).prev_sample

            # Return the first min(prediction_horizon, action_horizon) steps
            # This handles cases where prediction_horizon < action_horizon
            return_seq_len = min(self.prediction_horizon, self.action_horizon)
            return naction[:, :return_seq_len, :]

    def serialize(self):
        """
        Serialize model and EMA weights for checkpointing.
        """
        result = {
            "nets": self.nets.state_dict(),
        }
        if self.ema is not None:
            # Save EMA averaged model weights and EMA state
            result["ema"] = {
                "averaged_model": self.ema.averaged_model.state_dict(),
                "optimization_step": self.ema.optimization_step,
                "decay": self.ema.decay,
            }
        else:
            result["ema"] = None
        return result

    def deserialize(self, model_dict):
        """
        Load model and EMA weights from checkpoint.

        Supports:
        - Legacy checkpoints that only contain policy weights (no 'nets' key)
        - New-style checkpoints with separate 'nets' and 'ema' entries
        """
        # Legacy format: a flat state dict for self.nets["policy"]
        if "nets" not in model_dict and "policy" in model_dict:
            self.nets.load_state_dict(model_dict)
            return "Loaded legacy diffusion model"

        # New format: separate 'nets' and optional 'ema'
        if "nets" in model_dict:
            self.nets.load_state_dict(model_dict["nets"])

        if "ema" in model_dict and model_dict["ema"] is not None and self.ema is not None:
            print("Loading EMA state...")
            ema_state = model_dict["ema"]
            # Load averaged model weights
            if isinstance(ema_state, dict) and "averaged_model" in ema_state:
                self.ema.averaged_model.load_state_dict(ema_state["averaged_model"])
                # Restore EMA state if available
                if "optimization_step" in ema_state:
                    self.ema.optimization_step = ema_state["optimization_step"]
                if "decay" in ema_state:
                    self.ema.decay = ema_state["decay"]
            else:
                # Legacy format: assume ema_state is the averaged_model state_dict directly
                self.ema.averaged_model.load_state_dict(ema_state)

        return "Loaded diffusion model weights + EMA"


# ============================================================================
# Flow Matching Policy (The pi_0/0.5/*0.6 style flow matching policy)
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    1-D sinusoidal position encoding for Diffusion/Flow Matching timesteps.
    The PositionEmbeddingSine in DETRTransformer is generally 2D (H, W), which is not suitable for timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ActionHead(nn.Module):
    """
    DETR-style Action Expert for the flow matching policy.
    """

    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config

        # Helper function to get config value (supports both dict and object)
        def get_config(key, default=None):
            if isinstance(config, dict):
                value = config.get(key, default)
            else:
                value = getattr(config, key, default)
            # If value is None, return default
            return value if value is not None else default

        # config parameters
        hidden_dim = get_config("hidden_dim")
        action_dim = get_config("action_dim")
        self.seq_len = get_config("action_seq_len")

        # Build the Transformer directly using config (same as ACT)
        # build_transformer expects an object with attributes, so convert dict to object if needed
        from additional_modules.DETRTransformer import build_transformer
        from types import SimpleNamespace
        
        # Convert config to dict, handling both dict and object types
        if isinstance(config, dict):
            config_dict = config.copy()
        else:
            # Convert object to dict using vars() or __dict__
            config_dict = vars(config) if hasattr(config, '__dict__') else {}
            # Fallback: manually extract common attributes
            if not config_dict:
                for attr in ['hidden_dim', 'dim_feedforward', 'enc_layers_num', 'dec_layers', 
                            'nheads', 'dropout', 'pre_norm',
                            'use_gated_attention', 'mlp_type', 'gate_mode']:
                    if hasattr(config, attr):
                        config_dict[attr] = getattr(config, attr)
        
        # Set defaults for transformer config
        config_dict.setdefault("dropout", 0.1)
        config_dict.setdefault("nheads", get_config("n_heads", 8))
        config_dict.setdefault("enc_layers_num", 4)
        
        # Convert to SimpleNamespace for build_transformer
        transformer_config = SimpleNamespace(**config_dict)
        self.transformer = build_transformer(transformer_config)

        # Project the observation features to the hidden dimension
        obs_dim = get_config("obs_dim")
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)

        # Project the action features to the hidden dimension
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Re-project the hidden dimension to the action dimension (predict Vector Field)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        # Timestep embedding
        # Use SiLU (Swish) activation, commonly used in Diffusion/Flow Matching models
        # and consistent with EfficientNet backbone in this codebase
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),  # SiLU is more standard for generative models than Mish
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Sequence Positional Embedding
        # For distinguishing different time steps in the action sequence
        self.action_pos_embed = nn.Embedding(self.seq_len, hidden_dim)
    
    def forward(
        self,
        obs_embed,
        noisy_action,
        timestep
    ):
        """
        Args:
            obs_emb: [bs, N, obs_dim] (Image Patch Features) or [bs, obs_dim] (Global Features)
            noisy_action: [bs, seq, action_dim] (action with noise)
            timesteps: [bs] (Timesteps)
        """
        bs, seq, _ = noisy_action.shape

        # --------------------- input processing ---------------------
        # preparing the observation features (Memory/Encoder features)
        if len(obs_embed.shape) == 2:
            obs_embed = obs_embed.unsqueeze(1) # [bs, 1, obs_dim]
        src = self.obs_proj(obs_embed) # [bs, 1, hidden_dim]

        # DETR generally need the (seq, bs, dim) format for the memory features
        src = src.permute(1, 0, 2) # [1, bs, hidden_dim]

        # preparing the action Queries (Target / Decoder inputs)
        tgt = self.action_proj(noisy_action) # [bs, seq, hidden_dim]

        # time conditioning
        time_emb = self.time_mlp(timestep) # [bs, hidden_dim]

        # sequence positional embedding
        tgt = tgt + time_emb.unsqueeze(1) # [bs, seq, hidden_dim]

        tgt = tgt.permute(1, 0, 2) # [seq, bs, hidden_dim]

        # --------------------- Positional Embeddings ---------------------
        # Query Positional Embedding (slice to match actual sequence length)
        action_pos_embed = self.action_pos_embed.weight[:seq].unsqueeze(1).repeat(1, bs, 1) # [seq, bs, hidden_dim]

        # Src Positional Embedding
        src_pos = torch.zeros_like(src)

        # --------------------- Transformer Forward Pass ---------------------
        # Encoder: observation features
        memory = self.transformer.encoder(
            src,
            pos = src_pos,
        )

        # Decoder: action features
        hs = self.transformer.decoder(
            tgt,
            memory,
            pos = src_pos,
            query_pos = action_pos_embed,
        )

        # hs is generally a list of intermediate outputs from each layer
        # we take the last layer output for the final prediction
        hs = hs[-1].permute(1, 0, 2) # [bs, seq, hidden_dim] (permute from [seq, bs, hidden_dim])

        # --------------------- Action Prediction ---------------------
        # Predict the Vector Field
        action_pred = self.action_head(hs) # [bs, seq, action_dim]

        return action_pred

class FlowMatchingPolicy(nn.Module):
    """
    Flow Matching Policy for the flow matching training.
    Compatible with train.py interface.
    """
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.config = config
        
        # Helper function to get config value (supports both dict and object)
        def get_config(key, default=None):
            if isinstance(config, dict):
                return config.get(key, default)
            else:
                return getattr(config, key, default)
        
        # Extract config parameters (config can be dict or object)
        self.camera_names = get_config("camera_names")
        self.action_seq_len = get_config("action_seq_len")
        self.num_queries = self.action_seq_len  # For compatibility with train.py
        self.action_dim = get_config("action_dim")
        self.lr = get_config("lr")
        self.weight_decay = get_config("weight_decay", 1e-4)
        self.use_language = get_config("use_language", False)
        self.use_state = get_config("use_state", True)
        
        # Build observation feature extractor (similar to DiffusionPolicy)
        from llp.backbone_impl import FilMedBackbone
        from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax
        
        backbone_name = get_config("backbone", "efficientnet_b3film")
        use_language = get_config("use_language", False)
        
        # Get image size
        img_h = get_config("image_height", None)
        img_w = get_config("image_width", None)
        if img_h is not None and img_w is not None:
            pass
        elif get_config("image_size", None) is not None:
            image_size = get_config("image_size")
            if isinstance(image_size, int):
                img_h = img_w = image_size
            elif isinstance(image_size, (tuple, list)):
                img_h, img_w = image_size
            else:
                img_h = img_w = 224
        else:
            img_h = img_w = 224
        
        # Auto-detect backbone output shape
        dummy_input = torch.randn(1, 3, img_h, img_w)
        if use_language:
            temp_backbone = FilMedBackbone(backbone_name)
            dummy_cmd = torch.randn(1, 768)
            with torch.no_grad():
                dummy_feature = temp_backbone(dummy_input, dummy_cmd)
        else:
            temp_backbone = ResNet18Conv(
                input_channel=3,
                pretrained=False,
                input_coord_conv=False,
            )
            with torch.no_grad():
                dummy_feature = temp_backbone(dummy_input)
        
        _, c_out, h_out, w_out = dummy_feature.shape
        input_shape = (c_out, h_out, w_out)
        print(f"Policy Type: Flow Matching")
        print(f"Using FiLMed Backbone: {use_language}")
        if use_language:
            print(f"Backbone type: {backbone_name}")
        print(f"Input size: {img_h}x{img_w}")
        print(f"Output shape: {input_shape}")
        del temp_backbone
        
        # Build backbones, pools, and linears for each camera
        self.num_kp = 32
        self.feature_dim = 64
        backbones = []
        pools = []
        linears = []
        
        for _ in self.camera_names:
            if use_language:
                backbone = FilMedBackbone(backbone_name)
            else:
                backbone = ResNet18Conv(
                    input_channel=3,
                    pretrained=False,
                    input_coord_conv=False,
                )
            backbones.append(backbone)
            
            pools.append(
                SpatialSoftmax(
                    input_shape=input_shape,
                    num_kp=self.num_kp,
                    temperature=1.0,
                    learnable_temperature=False,
                    noise_std=0.0,
                )
            )
            
            linears.append(
                nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dim)
            )
        
        self.backbones = nn.ModuleList(backbones)
        self.pools = nn.ModuleList(pools)
        self.linears = nn.ModuleList(linears)
        
        # Calculate observation dimension
        # Only include state_dim if use_state is True
        state_dim = get_config("state_dim", 20)
        self.observation_dim = self.feature_dim * len(self.camera_names) + (state_dim if self.use_state else 0)
        
        if use_language:
            self.lang_embed_dim = 768
            self.lang_embed_proj = nn.Linear(self.lang_embed_dim, self.feature_dim)
            self.observation_dim += self.feature_dim
        
        # action generation expert
        # Update config with observation dimension (create a copy to avoid modifying original)
        if isinstance(config, dict):
            config_with_obs = config.copy()
            config_with_obs["obs_dim"] = self.observation_dim
        else:
            # If config is an object, create a dict-like wrapper
            class ConfigWrapper:
                def __init__(self, original_config, obs_dim):
                    self._original = original_config
                    self.obs_dim = obs_dim
                def __getattr__(self, key):
                    if key == "obs_dim":
                        return self.obs_dim
                    return getattr(self._original, key)
                def get(self, key, default=None):
                    if key == "obs_dim":
                        return self.obs_dim
                    return getattr(self._original, key, default)
            config_with_obs = ConfigWrapper(config, self.observation_dim)
        
        self.model = ActionHead(config_with_obs)
        
        # flow scheduler
        from .flow_scheduler import FlowMatchEulerDiscreteScheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps = get_config("num_train_timesteps", 1000),
            shift = get_config("shift", 1.0),
        )

        # paper parameters from Pi_0.5
        # alpha = 1.5, beta = 1.0, threshold = 0.999
        self.beta_dist = Beta(torch.tensor(1.5), torch.tensor(1.0))
        self.s_threshold = 0.999
        
        # Inference timesteps
        self.num_inference_timesteps = get_config("num_inference_timesteps", 10)

    def sample_timesteps(
        self,
        batch_size,
        device
    ):
        """
        Sample timesteps from the beta distribution: p(tau) ~ Beta((s-tau)/s; 1.5, 1.0).
        Note: if x = (s - tau) / s, then x ~ Beta(1.5, 1.0)
        Then, tau = s * (1 - x)
        s * x = s - tau
        tau = s * (1 - x)
        Args:
            batch_size: int
            device: str
        """
        # sample x from Beta(1.5, 1.0)
        x = self.beta_dist.sample((batch_size,)).to(device)
        # convert to tau (timestep): tau = s * (1 - x)
        # The result will prefer 0-end, which emphasizes the explicit training in low-noise region.
        tau = self.s_threshold * (1.0 - x)
        return tau

    def _extract_obs_embed(self, qpos, image, command_embedding=None):
        """
        Extract observation embeddings from images and qpos.
        Similar to DiffusionPolicy's observation extraction.
        """
        all_features = []
        
        # Check if lang_embed_proj exists (consistent with DiffusionPolicy)
        has_lang_proj = hasattr(self, 'lang_embed_proj')
        
        for cam_id, _ in enumerate(self.camera_names):
            cam_image = image[:, cam_id]
            
            # Consistent with DiffusionPolicy: check lang_embed_proj existence
            if has_lang_proj:
                cam_features = self.backbones[cam_id](cam_image, command_embedding)
            else:
                cam_features = self.backbones[cam_id](cam_image)
            
            pool_features = self.pools[cam_id](cam_features)
            pool_features = torch.flatten(pool_features, start_dim=1)
            out_features = self.linears[cam_id](pool_features)
            all_features.append(out_features)
        
        # Concatenate features
        # Only concatenate qpos if use_state is True
        features_to_concat = all_features.copy()
        if self.use_state:
            features_to_concat.append(qpos)
        
        if command_embedding is not None and has_lang_proj:
            command_embedding_proj = self.lang_embed_proj(command_embedding)
            features_to_concat.append(command_embedding_proj)
        
        obs_embed = torch.cat(features_to_concat, dim=1)
        
        # Debug: Verify observation dimension matches expected
        actual_obs_dim = obs_embed.shape[1]
        expected_obs_dim = self.observation_dim
        if actual_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"Observation dimension mismatch! "
                f"Actual: {actual_obs_dim}, Expected: {expected_obs_dim}. "
                f"This suggests a bug in observation_dim calculation. "
                f"all_features dims: {[f.shape[1] for f in all_features]}, "
                f"qpos dim: {qpos.shape[1] if self.use_state else 'N/A (use_state=False)'}, "
                f"use_state: {self.use_state}, "
                f"use_language: {command_embedding is not None and has_lang_proj}"
            )
        
        return obs_embed
    
    def __call__(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        command_embedding=None,
        encode_command=False,  # For interface compatibility (not used)
        **kwargs
    ):
        """
        Forward pass compatible with ACT/Diffusion interface.
        
        Args:
            qpos: [bs, state_dim] robot state
            image: [bs, num_cameras, C, H, W] images
            actions: [bs, seq_len, action_dim] ground truth actions (training) or None (inference)
            is_pad: [bs, seq_len] padding mask (for compatibility, not used in flow matching)
            command_embedding: [bs, lang_embed_dim] language embedding (optional)
            encode_command: Ignored (for interface compatibility)
        """
        # Extract observation embeddings
        obs_embed = self._extract_obs_embed(qpos, image, command_embedding)
        
        if actions is not None:
            # Training mode
            return self.compute_loss({
                "obs_embed": obs_embed,
                "actions": actions,
                "is_pad": is_pad
            })
        else:
            # Inference mode
            return self.get_action(obs_embed, self.num_inference_timesteps)
    
    def compute_loss(
        self,
        batch,
    ):
        obs_embed = batch["obs_embed"]
        x_0 = batch['actions'] # GT Action (Data) [bs, seq, action_dim]
        is_pad = batch.get('is_pad', None)  # [bs, seq] padding mask
        bs = x_0.shape[0]

        # ------------------------ beta sampled timesteps ------------------------
        tau  = self.sample_timesteps(bs, x_0.device)

        # ------------------------ flow matching process ------------------------
        # sample noise (use Gaussian distribution to match inference)
        x_1 = torch.randn_like(x_0) # [bs, seq, action_dim]
        # Constructing intermediate states x_t (rectified flow)
        # x_tau = tau * x_1 + (1 - tau) * x_0
        # tau=0 -> x_0 (data), tau=1 -> x_1 (noise)
        tau_expanded = tau.view(bs, 1, 1)
        x_tau = tau_expanded * x_1 + (1.0 - tau_expanded) * x_0

        # ------------------------ Calculate Target Vector Field ------------------------
        # flow u_t = x_1 - x_0 (direction to noise)
        # Then, we can start from noise, and flow to data in the -u_t direction during the inference.
        target_vfield = x_1 - x_0

        # ------------------------ Predict Vector Field ------------------------
        # project the tau to the hidden dimension range.
        model_timesteps = tau * self.scheduler.num_train_timesteps

        # Ensure the observation features is just took as the key/value in the cross-attention..
        pred_vfield = self.model(obs_embed, x_tau, model_timesteps)

        # ------------------------ Calculate Loss ------------------------
        # Handle padding mask similar to diffusion policy
        all_loss = F.mse_loss(pred_vfield, target_vfield, reduction="none")  # [bs, seq, action_dim]
        
        if is_pad is not None:
            # Apply padding mask: is_pad=True means padding, so we mask it out
            mask = (~is_pad).unsqueeze(-1)  # [batch, seq, 1]
            # valid should count all valid action dimensions, not just time steps
            action_dim = x_0.shape[-1]
            valid = mask.sum() * action_dim  # total number of valid (non-padded) action elements
            loss = (all_loss * mask).sum() / valid.clamp(min=1)
        else:
            loss = all_loss.mean()

        return {
            "loss": loss,
        }

    @torch.no_grad()
    def get_action(
        self,
        obs_embed,
        num_inference_steps = None
    ):
        """
        Keep Euler Step in inference process. 
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_timesteps
            
        bs = obs_embed.shape[0]
        action_dim = self.action_dim
        seq_len = self.action_seq_len
        device = obs_embed.device

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        x_t = torch.randn(bs, seq_len, action_dim, device=device) # [bs, seq, action_dim]

        # Iterate through timesteps (from noise t=1.0 to data t=0.0)
        # Note: timesteps has num_inference_steps + 1 elements, we exclude the last one (t=0.0)
        # because we need num_inference_steps steps to go from t=1.0 to t=0.0
        for i, t in enumerate(self.scheduler.timesteps[:-1]):
            # t is already scaled to [0, num_train_timesteps] in set_timesteps
            # Expand timestep to batch dimension for time_mlp
            if isinstance(t, torch.Tensor):
                t_batch = t.expand(bs) if t.numel() == 1 else t
            else:
                t_batch = torch.full((bs,), t, device=device, dtype=torch.float32)
            
            model_output = self.model(obs_embed, x_t, t_batch)

            # Euler Step: x_{t-1} = x_t + (sigma_next - sigma_curr) * v
            # Use index directly to avoid searching (more efficient)
            step_output = self.scheduler.step(model_output, t, x_t, step_index=i)
            x_t = step_output[0]

        return x_t
    
    def configure_optimizers(self):
        """Configure optimizer for training."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
    
    def serialize(self):
        """Serialize model state for checkpointing."""
        return self.state_dict()
    
    def deserialize(self, model_dict, strict=True):
        """Load model state from checkpoint."""
        self.load_state_dict(model_dict, strict=strict)
        return "Loaded flow matching model"
