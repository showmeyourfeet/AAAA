import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from llp.backbone_impl import FilMedBackbone

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override: dict) -> None:
        super().__init__()
        from robomimic.models.base_nets import ResNet18Conv, ResNet50Conv, SpatialSoftmax
        from robomimic.algo.diffusion_policy import (
            replace_bn_with_gn,
            ConditionalUnet1D,
        )
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler


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

        img_h = args_override.get("image_height", 540)
        img_w = args_override.get("image_width", 360)

        if "image_size" in args_override:
            image_size = args_override["image_size"]
            if isinstance(image_size, int):
                img_h = img_w = image_size
            elif isinstance(image_size, (tuple, list)):
                img_h, img_w = image_size

        dummy_input = torch.randn(1, 3, img_h, img_w)
        if use_language:
            print(f"Using FiLMed backbone: {backbone_name}")
            temp_backbone = FilMedBackbone(backbone_name)
            dummy_cmd = torch.randn(1, 768)
            with torch.no_grad():
                dummy_feature = temp_backbone(dummy_input, dummy_cmd)
        else:
            print(f"Using Robomimic ResNet18Conv backbone.")
            temp_backbone = ResNet18Conv(**{
                "input_channel": 3,
                "pretrained": False,
                "input_coord_conv": False,
            })
            with torch.no_grad():
                dummy_feature = temp_backbone(dummy_input)

        _, c_out, h_out, w_out = dummy_feature.shape
        input_shape = (c_out, h_out, w_out)
        print(f"Auto-detected backbone output shape: {input_shape} from input {img_h}x{img_w}")

        del temp_backbone

        backbones = []
        pools = []
        linears = []

        for _ in self.camera_names:
            if use_language:
                print(f"Using FiLMed Backbone: {backbone_name}")
                # initialize the FiLMed backbone
                backbone = FilMedBackbone(backbone_name)
            else:
                print(f"Using Robomimic ResNet18Conv backbone.")
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
        self.backbones = replace_bn_with_gn(self.backbones)

        # calculate Observation Dimension
        state_dim = args_override.get("state_dim", 20)
        self.observation_dim = self.feature_dim * len(self.camera_names) + state_dim
        
        # 定义语言嵌入维度
        self.lang_embed_dim = 768

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

        nets = self.nets.float() # update reference

        # Modified: 删除了内部 DataParallel，因为它破坏了字典结构，多卡请在外部处理
        # if args_override.get("multi_gpu", False):
        #     nets = nn.DataParallel(nets)
        
        device = args_override.get("device", "cuda")
        self.nets.to(device)
        # self.nets = nets # unnecessary since we used self.nets.to()

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps = 100,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

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

        if command_embedding is not None and "lang_embed_proj" in nets["policy"]:
            command_embedding_proj = nets["policy"]["lang_embed_proj"](command_embedding)
            observation_condition = torch.cat(all_features + [qpos] + [command_embedding_proj], dim=1)
        else:
            observation_condition = torch.cat(all_features + [qpos], dim=1)
        
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
            
            # Also calculate L1 loss for compatibility with train.py
            all_l1 = F.l1_loss(
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
                l1_loss = (all_l1 * mask).sum() / valid.clamp(min=1)
            else:
                loss = all_l2.mean()
                l1_loss = all_l1.mean()

            return {
                "l2_loss": loss,
                "l1": l1_loss,  # Added for compatibility with train.py
                "loss": loss
            }
        else:
            # ------------------------------------------- Inference Mode -------------------------------------------
            # init action (Gaussian Prior)
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

            # Return only the first action_horizon steps for compatibility
            # (prediction_horizon may be larger than action_horizon)
            return naction[:, :self.action_horizon, :]

    def serialize(self):
        return self.nets.state_dict()

    def deserialize(self, model_dict):
            # Load model state dict.
            if "nets" in model_dict:
                self.nets.load_state_dict(model_dict["nets"])
            else:
                self.nets.load_state_dict(model_dict)
            return "Loaded diffusion model weights"