"""ACT model construction utilities."""

from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from .DETRTransformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    build_transformer,
)
from .backbone_impl import build_backbone, FilMedBackbone, FrozenBatchNorm2d


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    def get_position_angle_vec(position: int):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """Variant of DETR used for ACT."""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        use_language=False,
        use_film=False,
        shared_backbone=False,
        num_command=2,
        vq=False,
        vq_class=512,
        vq_dim=32,
        use_state: bool = True,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        self.vq, self.vq_class, self.vq_dim = vq, vq_class, vq_dim
        self.use_state = use_state
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.use_language = use_language
        self.use_film = use_film
        self.shared_backbone = shared_backbone
        if use_language:
            self.lang_embed_proj = nn.Linear(768, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = (
                nn.Linear(20, hidden_dim) if self.use_state else None
            )
        else:
            self.input_proj_robot_state = (
                nn.Linear(20, hidden_dim) if self.use_state else None
            )
            self.input_proj_env_state = nn.Linear(10, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(20, hidden_dim)  # project action to embedding
        self.encoder_joint_proj = (
            nn.Linear(20, hidden_dim) if self.use_state else None  # project qpos to embedding
        )
        
        print(f"Use VQ: {self.vq}, {self.vq_class}, {self.vq_dim}")
        if self.vq:
            self.latent_proj = nn.Linear(hidden_dim, self.vq_class * self.vq_dim)
        else:
            self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + (1 if self.use_state else 0) + num_queries, hidden_dim),
            persistent=False,
        )

        # decoder extra parameters
        if self.vq:
            self.latent_out_proj = nn.Linear(self.vq_class * self.vq_dim, hidden_dim)
        else:
            self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        
        pos_embed_dim = 1  # latent token always present
        self.latent_pos_id = 0
        if self.use_state:
            self.proprio_pos_id = pos_embed_dim
            pos_embed_dim += 1
        else:
            self.proprio_pos_id = None
        if self.use_language:
            self.command_pos_id = pos_embed_dim
            pos_embed_dim += 1
        else:
            self.command_pos_id = None
        self.additional_pos_embed = nn.Embedding(pos_embed_dim, hidden_dim)

    def _get_additional_pos_embed(
        self, include_proprio: bool, include_command: bool
    ) -> torch.Tensor:
        embed_ids = [self.latent_pos_id]
        if include_proprio and self.proprio_pos_id is not None:
            embed_ids.append(self.proprio_pos_id)
        if include_command and self.command_pos_id is not None:
            embed_ids.append(self.command_pos_id)
        return self.additional_pos_embed.weight[embed_ids]

    def encode(self, qpos, actions=None, is_pad=None, vq_sample=None):
        """
        Encode action sequence to latent representation.
        
        Args:
            qpos: batch, qpos_dim
            actions: batch, seq, action_dim (training only)
            is_pad: batch, seq (training only)
            vq_sample: VQ sample for inference (optional)
        
        Returns:
            latent_input: batch, hidden_dim
            probs: batch, vq_class, vq_dim (VQ mode) or None (VAE mode)
            binaries: batch, vq_class, vq_dim (VQ mode) or None (VAE mode)
            mu: batch, latent_dim (VAE mode) or None (VQ mode)
            logvar: batch, latent_dim (VAE mode) or None (VQ mode)
        """
        bs, _ = qpos.shape
        if self.encoder is None:
            # No encoder: use zero latent vector
            if self.vq:
                vq_zero = torch.zeros([bs, self.vq_class * self.vq_dim], dtype=torch.float32, device=qpos.device)
                latent_input = self.latent_out_proj(vq_zero)
            else:
                latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
                latent_input = self.latent_out_proj(latent_sample)
            probs = binaries = mu = logvar = None
        else:
            # CVAE encoder
            is_training = actions is not None  # train or val
            if is_training:
                # Adapt to dataset action dimension dynamically
                act_dim = actions.shape[-1]
                if self.encoder_action_proj.in_features != act_dim:
                    self.encoder_action_proj = nn.Linear(act_dim, self.hidden_dim).to(actions.device)
                if self.action_head.out_features != act_dim:
                    self.action_head = nn.Linear(self.hidden_dim, act_dim).to(qpos.device)
                # Project action sequence to embedding dim
                action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
                
                cls_embed = self.cls_embed.weight  # (1, hidden_dim)
                cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)

                prefix_tokens = [cls_embed]
                prefix_pad = [torch.full((bs, 1), False, device=cls_embed.device)]

                if self.use_state and qpos is not None:
                    q_dim = qpos.shape[-1]
                    if self.encoder_joint_proj is None or self.encoder_joint_proj.in_features != q_dim:
                        self.encoder_joint_proj = nn.Linear(q_dim, self.hidden_dim).to(qpos.device)
                        if self.input_proj_robot_state is not None:
                            self.input_proj_robot_state = nn.Linear(q_dim, self.hidden_dim).to(qpos.device)
                    qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
                    qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
                    prefix_tokens.append(qpos_embed)
                    prefix_pad.append(torch.full((bs, 1), False, device=qpos.device))

                encoder_input = torch.cat(prefix_tokens + [action_embed], axis=1)
                encoder_input = encoder_input.permute(1, 0, 2)

                cls_joint_is_pad = torch.cat(prefix_pad, axis=1)
                is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)
                
                # Obtain position embedding
                pos_embed = self.pos_table.clone().detach()
                pos_embed = pos_embed.permute(1, 0, 2)  # (seq+2, 1, hidden_dim)
                
                # Query model
                encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
                encoder_output = encoder_output[0]  # take cls output only
                latent_info = self.latent_proj(encoder_output)
                
                if self.vq:
                    # Vector Quantization: discrete latent representation
                    # latent_info: (bs, vq_class * vq_dim)
                    logits = latent_info.reshape([*latent_info.shape[:-1], self.vq_class, self.vq_dim])
                    # logits: (bs, vq_class, vq_dim)
                    probs = torch.softmax(logits, dim=-1)
                    # probs: (bs, vq_class, vq_dim)
                    # Sample from each vq_class position independently
                    binaries = F.one_hot(
                        torch.multinomial(probs.view(-1, self.vq_dim), 1).squeeze(-1),
                        self.vq_dim
                    ).view(-1, self.vq_class, self.vq_dim).float()
                    # binaries: (bs, vq_class, vq_dim)
                    binaries_flat = binaries.view(-1, self.vq_class * self.vq_dim)
                    # binaries_flat: (bs, vq_class * vq_dim)
                    probs_flat = probs.view(-1, self.vq_class * self.vq_dim)
                    # probs_flat: (bs, vq_class * vq_dim)
                    # Straight-through estimator: forward pass uses discrete, backward pass uses continuous
                    straight_through = binaries_flat - probs_flat.detach() + probs_flat
                    # straight_through: (bs, vq_class * vq_dim)
                    latent_input = self.latent_out_proj(straight_through)
                    # latent_input: (bs, hidden_dim)
                    mu = logvar = None
                else:
                    # Standard VAE: continuous latent representation
                    probs = binaries = None
                    mu = latent_info[:, : self.latent_dim]
                    logvar = latent_info[:, self.latent_dim :]
                    latent_sample = reparametrize(mu, logvar)
                    latent_input = self.latent_out_proj(latent_sample)
            else:
                # Inference
                mu = logvar = binaries = probs = None
                if self.vq:
                    # Use provided vq_sample if available, otherwise use zero
                    if vq_sample is not None:
                        latent_input = self.latent_out_proj(vq_sample.view(-1, self.vq_class * self.vq_dim))
                    else:
                        # Use zero vector for VQ
                        vq_zero = torch.zeros([bs, self.vq_class * self.vq_dim], dtype=torch.float32, device=qpos.device)
                        latent_input = self.latent_out_proj(vq_zero)
                else:
                    # Standard VAE: use zero latent vector
                    latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32, device=qpos.device)
                    latent_input = self.latent_out_proj(latent_sample)

        return latent_input, probs, binaries, mu, logvar

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        command_embedding=None,
        vq_sample=None,
    ):
        """
        Forward pass.
        
        Args:
            qpos: batch, qpos_dim
            image: batch, num_cam, channel, height, width
            env_state: None
            actions: batch, seq, action_dim (training only)
            is_pad: batch, seq (training only)
            command_embedding: batch, command_embedding_dim (optional)
            vq_sample: VQ sample (not used in our implementation)
        
        Returns:
            a_hat: predicted actions
            is_pad_hat: predicted padding
            [mu, logvar]: VAE latent parameters
            probs: None (VQ not used)
            binaries: None (VQ not used)
        """
        # Encode action sequence to latent
        latent_input, probs, binaries, mu, logvar = self.encode(qpos, actions, is_pad, vq_sample)

        # Project command embedding if using language
        if command_embedding is not None and self.use_language:
            command_embedding_proj = self.lang_embed_proj(command_embedding)
        else:
            command_embedding_proj = None

        if self.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            for cam_id, _ in enumerate(self.camera_names):
                backbone_idx = 0 if self.shared_backbone else cam_id
                if self.use_film:
                    features, pos = self.backbones[backbone_idx](image[:, cam_id], command_embedding)
                else:
                    features, pos = self.backbones[backbone_idx](image[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)

            proprio_input = None
            if self.use_state:
                if self.input_proj_robot_state is None or self.input_proj_robot_state.in_features != qpos.shape[-1]:
                    self.input_proj_robot_state = nn.Linear(qpos.shape[-1], self.hidden_dim).to(qpos.device)
                proprio_input = self.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            command_embedding_to_append = command_embedding_proj if self.use_language else None
            additional_pos_embed = self._get_additional_pos_embed(
                include_proprio=proprio_input is not None,
                include_command=command_embedding_to_append is not None,
            )
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                additional_pos_embed,
                command_embedding=command_embedding_to_append,
            )[-1] # take the last layer output
        else:
            qpos_proj = None
            if self.use_state:
                if self.input_proj_robot_state is None or self.input_proj_robot_state.in_features != qpos.shape[-1]:
                    self.input_proj_robot_state = nn.Linear(qpos.shape[-1], self.hidden_dim).to(qpos.device)
                qpos_proj = self.input_proj_robot_state(qpos)
            env_state_proj = self.input_proj_env_state(env_state)
            transformer_input = (
                torch.cat([qpos_proj, env_state_proj], axis=1) if qpos_proj is not None else env_state_proj
            )
            hs = self.transformer(
                transformer_input,
                None,
                self.query_embed.weight,
                self.pos.weight,
            )[-1] # take the last layer output
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar], probs, binaries


def build_encoder(args) -> TransformerEncoder:
    encoder_layer = TransformerEncoderLayer(
        args.hidden_dim,
        args.nheads,
        args.dim_feedforward,
        args.dropout,
        "relu",
        args.pre_norm,
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    encoder = TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)
    return encoder


def build_act_model(args) -> DETRVAE:
    state_dim = getattr(args, "state_dim", 20)

    backbones = []
    share_backbone = not args.use_language and "film" not in args.backbone

    if share_backbone:
        backbones.append(build_backbone(args))
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)

    transformer = build_transformer(args)

    if args.no_encoder:
        encoder = None
    else:
        encoder = build_encoder(args)

    vq = getattr(args, "vq", False)
    vq_class = getattr(args, "vq_class", 512)
    vq_dim = getattr(args, "vq_dim", 32)
    
    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        use_language=args.use_language,
        use_film="film" in args.backbone,
        vq=vq,
        vq_class=vq_class,
        vq_dim=vq_dim,
        shared_backbone=share_backbone,
        use_state=getattr(args, "use_state", True),
    )

    # Handle image encoder training control
    train_image_encoder = getattr(args, "train_image_encoder", False)
    if not train_image_encoder:
        # Freeze all backbone parameters
        for backbone in model.backbones:
            for param in backbone.parameters():
                param.requires_grad = False
        print("Image encoder backbones frozen (default)")
    else:
        print("Image encoder backbones unfrozen for training")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("ACT model configuration", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--clip_max_norm", default=0.1, type=float)

    parser.add_argument("--backbone", default="resnet18", type=str)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned", "v2", "v3"))
    parser.add_argument("--camera_names", default=[], type=list)

    parser.add_argument("--enc_layers", default=4, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=400, type=int)
    parser.add_argument("--pre_norm", action="store_true")

    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--use_language", action="store_true")
    parser.add_argument("--state_dim", default=20, type=int)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no_encoder", action="store_true", help="Disable VAE encoder, use zero latent vector instead")
    parser.add_argument("--vq", action="store_true", help="Use Vector Quantization instead of standard VAE")
    parser.add_argument("--vq_class", type=int, default=512, help="Number of VQ codebook classes")
    parser.add_argument("--vq_dim", type=int, default=32, help="Dimension of each VQ codebook entry")

    return parser


def build_act_model_and_optimizer(args_override: dict):
    parser = argparse.ArgumentParser("ACT model", parents=[get_args_parser()])
    args = parser.parse_args([])

    for k, v in args_override.items():
        if v is not None:
            setattr(args, k, v)

    model = build_act_model(args)

    if args.multi_gpu and not args.eval:
        assert torch.cuda.device_count() > 1
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Build optimizer with separate learning rates for backbone and other parameters
    train_image_encoder = getattr(args, "train_image_encoder", False)
    non_backbone_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    
    if train_image_encoder and len(backbone_params) > 0:
        # Train backbone with separate learning rate
        param_dicts = [
            {"params": non_backbone_params},
            {"params": backbone_params, "lr": args.lr_backbone},
        ]
        print(f"Optimizer: backbone lr={args.lr_backbone}, other lr={args.lr}")
    else:
        # Backbone frozen or no backbone params
        param_dicts = [{"params": non_backbone_params}]
        print(f"Optimizer: only non-backbone parameters, lr={args.lr}")

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return model, optimizer
