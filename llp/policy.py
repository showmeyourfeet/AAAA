"""Policy definitions for ACT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

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
    ):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
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
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _), _, _ = self.model(
                qpos, image, env_state, vq_sample=vq_sample, command_embedding=command_embedding
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
