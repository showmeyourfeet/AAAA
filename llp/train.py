"""Training entrypoint for ACT policies."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange

try:
    # When run as a module: python -m act_refactor.train
    from .dataset import (
        SplittedEpisodicDataset,
        load_merged_data,
        load_splitted_data,
    )
    from .policy import ACTPolicy
    from .utils import (
        compute_dict_mean,
        detach_dict,
        is_multi_gpu_checkpoint,
        set_seed,
    )
except ImportError:
    # When run as a script: python path/to/act_refactor/train.py
    import sys
    from os.path import dirname, abspath

    sys.path.append(dirname(dirname(abspath(__file__))))
    from llp.dataset import (
        SplittedEpisodicDataset,
        load_merged_data,
        load_splitted_data,
    )
    from llp.policy import ACTPolicy
    from llp.utils import (
        compute_dict_mean,
        detach_dict,
        is_multi_gpu_checkpoint,
        set_seed,
    )


def infer_state_dim_from_checkpoint(model_state_dict: Dict[str, torch.Tensor]):
    possible_keys = [
        "model.action_head.weight",
        "model.module.action_head.weight",
        "action_head.weight",
    ]

    for key in possible_keys:
        if key in model_state_dict:
            weight_shape = model_state_dict[key].shape
            if len(weight_shape) == 2:
                return weight_shape[0]

    for key in model_state_dict.keys():
        if "action_head" in key and "weight" in key:
            weight_shape = model_state_dict[key].shape
            if len(weight_shape) == 2:
                return weight_shape[0]
    return None


def make_policy(policy_config: Dict) -> ACTPolicy:
    policy = ACTPolicy(policy_config)
    return policy


def make_optimizer(policy: ACTPolicy) -> torch.optim.Optimizer:
    return policy.configure_optimizers()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def make_scheduler(optimizer: torch.optim.Optimizer, num_steps: int) -> LambdaLR:
    return get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_steps // 100, num_training_steps=num_steps
    )


def forward_pass(data, policy: ACTPolicy):
    if len(data) == 5:
        image_data, qpos_data, action_data, is_pad, command_embedding = data
        command_embedding = command_embedding.cuda()
    else:
        image_data, qpos_data, action_data, is_pad = data
        command_embedding = None
    image_data = image_data.cuda()
    qpos_data = qpos_data.cuda()
    action_data = action_data.cuda()
    is_pad = is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad, command_embedding)


def train_bc(
    train_dataloader,
    config: Dict,
    val_dataloader=None,
    logger: logging.Logger | None = None,
):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    log_every = config.get("log_every", 1)

    policy_config = config["policy_config"]

    set_seed(seed)

    load_ckpt = None
    ckpt_path = None
    checkpoint = None
    if os.path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) > 2:
        print(f"Checkpoint directory {ckpt_dir} is not empty. Load checkpoint? (y/n)")
        load_ckpt = input().strip().lower()
        if load_ckpt == "y":
            # Priority 1: Try to load latest checkpoint (rolling backup)
            latest_ckpt_path = os.path.join(ckpt_dir, f"policy_latest_seed_{seed}.ckpt")
            if os.path.exists(latest_ckpt_path):
                ckpt_path = latest_ckpt_path
                checkpoint = torch.load(ckpt_path)
                print(f"Found latest checkpoint: {ckpt_path}")
            else:
                # Priority 2: Fallback to policy_epoch_* files
                existing_epochs = [
                    int(f.split("_")[2])
                    for f in os.listdir(ckpt_dir)
                    if f.startswith("policy_epoch_")
                ]
                if existing_epochs:
                    latest_idx = max(existing_epochs)
                    ckpt_path = os.path.join(
                        ckpt_dir, f"policy_epoch_{latest_idx}_seed_{seed}.ckpt"
                    )
                    checkpoint = torch.load(ckpt_path)
                    print(f"Found periodic checkpoint: {ckpt_path}")
            
            if checkpoint is not None:
                model_state_dict = checkpoint["model_state_dict"]
                ckpt_state_dim = infer_state_dim_from_checkpoint(model_state_dict)
                if ckpt_state_dim is not None and ckpt_state_dim != policy_config.get("state_dim", 20):
                    print(
                        "Warning: checkpoint state_dim differs from config."
                        f" Updating to {ckpt_state_dim}."
                    )
                    policy_config["state_dim"] = ckpt_state_dim

    policy = make_policy(policy_config)
    optimizer = make_optimizer(policy)
    scheduler = make_scheduler(optimizer, num_epochs)

    if load_ckpt == "y" and checkpoint is not None:
        print(f"Loading checkpoint from {ckpt_path}")
        model_state_dict = checkpoint["model_state_dict"]
        multi_gpu = policy_config.get("multi_gpu", False)
        if multi_gpu and not is_multi_gpu_checkpoint(model_state_dict):
            model_state_dict = {
                k if "model" not in k else f"model.module.{k.split('.', 1)[1]}": v
                for k, v in model_state_dict.items()
            }
        elif not multi_gpu and is_multi_gpu_checkpoint(model_state_dict):
            model_state_dict = {
                k.replace("module.", "", 1): v for k, v in model_state_dict.items()
            }
        loading_status = policy.deserialize(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(loading_status)
    else:
        start_epoch = 0

    policy.cuda()
    best_val = float("inf")
    best_ckpt_path = None
    
    # Restore best_val from checkpoint if available
    if load_ckpt == "y" and checkpoint is not None and "best_val" in checkpoint:
        best_val = checkpoint["best_val"]
        print(f"Restored best_val: {best_val:.5f}")
    
    train_history: List[Dict[str, torch.Tensor]] = []

    epoch_bar = trange(start_epoch, num_epochs, desc="Epochs", leave=True)
    for epoch in epoch_bar:
        policy.train()
        optimizer.zero_grad()
        batch_bar = tqdm(train_dataloader, desc=f"Train {epoch}", leave=False)
        for batch_idx, data in enumerate(batch_bar):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            if "l1" in forward_dict:
                batch_bar.set_postfix(
                    {
                        "l1": f"{forward_dict['l1'].item():.4f}",
                        "loss": f"{loss.item():.4f}",
                    }
                )
        scheduler.step()
        e = epoch - start_epoch
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (e + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        msg = f"Train loss: {epoch_train_loss:.5f}"
        if logger:
            logger.info(msg)
        else:
            tqdm.write(msg)
        epoch_summary["lr"] = np.array(scheduler.get_last_lr()[0])
        if (epoch + 1) % log_every == 0:
            summary_string = " ".join(
                [f"{k}: {v.item():.5f}" for k, v in epoch_summary.items()]
            )
            if logger:
                logger.info(summary_string)
            else:
                tqdm.write(summary_string)

        if val_dataloader is not None:
            policy.eval()
            val_losses = []
            with torch.inference_mode():
                val_bar = tqdm(val_dataloader, desc=f"Val {epoch}", leave=False)
                for data in val_bar:
                    forward_dict = forward_pass(data, policy)
                    val_losses.append(forward_dict["loss"].item())
                    val_bar.set_postfix({"loss": f"{val_losses[-1]:.4f}"})
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if logger:
                logger.info(f"Val loss: {val_loss:.5f}")
            else:
                tqdm.write(f"Val loss: {val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                best_ckpt_path = os.path.join(ckpt_dir, f"policy_best_seed_{seed}.ckpt")
                torch.save(
                    {
                        "model_state_dict": policy.serialize(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val": best_val,
                    },
                    best_ckpt_path,
                )
                if logger:
                    logger.info(f"Saved best checkpoint to {best_ckpt_path}")

        # Save latest checkpoint every epoch (rolling backup)
        # This allows resuming from any epoch, not just multiples of 100
        latest_ckpt_path = os.path.join(ckpt_dir, f"policy_latest_seed_{seed}.ckpt")
        torch.save(
            {
                "model_state_dict": policy.serialize(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            },
            latest_ckpt_path,
        )

        save_ckpt_every = 100
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path_epoch = os.path.join(
                ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt"
            )
            torch.save(
                {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path_epoch,
            )
            # prune_epoch = epoch - save_ckpt_every
            # if prune_epoch % 1000 != 0:
            #     prune_path = os.path.join(
            #         ckpt_dir, f"policy_epoch_{prune_epoch}_seed_{seed}.ckpt"
            #     )
            #     if os.path.exists(prune_path):
            #         os.remove(prune_path)

    ckpt_path_final = os.path.join(ckpt_dir, "policy_last.ckpt")
    torch.save(
        {
            "model_state_dict": policy.serialize(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
        },
        ckpt_path_final,
    )
    if logger:
        logger.info(f"Saved final checkpoint to {ckpt_path_final}")


def main(args: Dict):
    set_seed(args["seed"])

    ckpt_dir = args["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = logging.getLogger("train")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler(os.path.join(ckpt_dir, "train.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    use_language = args["use_language"]
    language_encoder = args["language_encoder"]

    dataset_dirs: List[str] = []
    num_episodes_list: List[int] = []
    camera_names: Sequence[str] = []
    max_episode_len = 0

    use_splitted = args.get("use_splitted", False)
    splitted_root = args.get("splitted_root")
    stage_embeddings_file = args.get("stage_embeddings_file")

    if use_splitted:
        if not splitted_root:
            raise ValueError("--splitted_root must be provided when --use_splitted is set")
        camera_names = args.get("camera_names", ["left_frame", "right_frame"])
        max_episode_len = args.get("chunk_size", 10)
        train_dataloader, stats, _ = load_splitted_data(
            root_dir=splitted_root,
            camera_names=camera_names,
            batch_size_train=args["batch_size"],
            max_len=max_episode_len,
            use_language=use_language,
            stage_embeddings_file=stage_embeddings_file,
            use_augmentation=args.get("use_augmentation", False),
            image_size=args.get("image_size", 224),
        )
        val_dataloader = None
        val_root = args.get("val_splitted_root")
        if val_root:
            stage_embeddings = None
            if use_language and stage_embeddings_file and os.path.exists(stage_embeddings_file):
                with open(stage_embeddings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("stage_embeddings", data)
                stage_embeddings = {
                    int(e["stage"]): e["embedding"]
                    for e in entries
                    if "stage" in e and "embedding" in e
                }
            val_dataset = SplittedEpisodicDataset(
                val_root,
                camera_names,
                stats,
                max_len=max_episode_len,
                use_language=use_language,
                stage_embeddings=stage_embeddings,
                image_size=args.get("image_size", 224),
                use_augmentation=False,  # No augmentation for validation
                training=False,
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                pin_memory=True,
                num_workers=4,
                prefetch_factor=2,
                persistent_workers=False,
            )
    else:
        # Expect task_config list describing dataset directories
        task_configs: Sequence[Dict] = args.get("task_config", [])
        if not task_configs:
            raise ValueError("Either --use_splitted or --task_config must be provided")
        for task_config in task_configs:
            dataset_dirs.append(task_config["dataset_dir"])
            num_episodes_list.append(task_config["num_episodes"])
            max_episode_len = max(max_episode_len, task_config["episode_len"])
            camera_names = task_config["camera_names"]
        train_dataloader, stats, _ = load_merged_data(
            dataset_dirs,
            num_episodes_list,
            camera_names,
            args["batch_size"],
            max_len=max_episode_len,
            command_list=args.get("command_list", []),
            use_language=use_language,
            language_encoder=language_encoder,
            policy_class="ACT",
            use_augmentation=args.get("use_augmentation", False),
            image_size=args.get("image_size", 224),
        )
        val_dataloader = None

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    policy_config = {
        "lr": args["lr"],
        "num_queries": args["chunk_size"],
        "kl_weight": args["kl_weight"],
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr_backbone": args.get("lr_backbone", 1e-5),
        "backbone": args["image_encoder"],
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
        "multi_gpu": args["multi_gpu"],
        "use_language": use_language,
        "state_dim": stats.get("action_mean", np.zeros(0)).shape[0] if "action_mean" in stats else 20,
        "no_encoder": args.get("no_encoder", False),
        "vq": args.get("vq", False),
        "vq_class": args.get("vq_class", 512),
        "vq_dim": args.get("vq_dim", 32),
        "train_image_encoder": args.get("train_image_encoder", False),
    }

    config = {
        "num_epochs": args["num_epochs"],
        "ckpt_dir": ckpt_dir,
        "seed": args["seed"],
        "policy_config": policy_config,
        "log_every": args.get("log_every", 1),
    }

    train_bc(train_dataloader, config, val_dataloader=val_dataloader, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--image_encoder", type=str, default="efficientnet_b3film")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--no_encoder", action="store_true", help="Disable VAE encoder, use zero latent vector instead")
    parser.add_argument("--vq", action="store_true", help="Use Vector Quantization instead of standard VAE")
    parser.add_argument("--vq_class", type=int, default=512, help="Number of VQ codebook classes")
    parser.add_argument("--vq_dim", type=int, default=32, help="Dimension of each VQ codebook entry")

    parser.add_argument("--use_language", action="store_true")
    parser.add_argument("--language_encoder", type=str, default="distilbert", choices=["distilbert", "clip"])
    parser.add_argument("--command_list", nargs="*", default=[])

    parser.add_argument("--use_splitted", action="store_true")
    parser.add_argument("--splitted_root", type=str)
    parser.add_argument("--val_splitted_root", type=str)
    parser.add_argument("--stage_embeddings_file", type=str)
    parser.add_argument("--camera_names", nargs="*", default=["left_frame", "right_frame"])
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation for images")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for augmentation")

    parser.add_argument("--task_config", type=json.loads, default=None)
    parser.add_argument("--log_every", type=int, default=1)
    
    # Image encoder training control
    parser.add_argument("--train_image_encoder", action="store_true", help="Enable training of the image encoder (default: frozen)")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for image encoder backbone (only used if --train_image_encoder is set)")

    args = parser.parse_args()
    main(vars(args))
