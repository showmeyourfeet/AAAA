"""Training entrypoint for ACT policies."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import pickle
import re
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange
import math


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
        load_data,
        load_data_srt,
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
        load_data,
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


def count_parameters(model: torch.nn.Module):
    """
    Count the number of parameters in a model.
    
    Returns:
        tuple: (total_params, trainable_params, non_trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


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


def compute_splitted_norm_stats(roots: Sequence[str]) -> Dict[str, np.ndarray]:
    """聚合多个 splitted 根目录的统计量，生成 action/qpos 的均值和方差。"""
    all_actions, all_qpos = [], []
    for root_dir in roots:
        if not root_dir:
            continue
        for stage_dir in sorted(os.listdir(root_dir)):
            stage_path = os.path.join(root_dir, stage_dir)
            if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                continue
            for run_dir in sorted(os.listdir(stage_path)):
                run_path = os.path.join(stage_path, run_dir)
                if not (run_dir.startswith("run") and os.path.isdir(run_path)):
                    continue
                data_file = os.path.join(run_path, "data.txt")
                if not os.path.exists(data_file):
                    continue
                with open(data_file, "r", encoding="utf-8") as f:
                    content = f.read()
                frame_blocks = re.split(r"Frame_\d+:", content)[1:]
                q_list, a_list = [], []
                for block in frame_blocks:
                    m_act = re.search(r"action:\s*\[([^\]]+)\]", block)
                    m_lr = re.search(r"lrstate:\s*\[([^\]]+)\]", block)
                    if m_act and m_lr:
                        try:
                            a_list.append(
                                [float(x.strip()) for x in m_act.group(1).split(",")]
                            )
                            q_list.append(
                                [float(x.strip()) for x in m_lr.group(1).split(",")]
                            )
                        except Exception:
                            continue
                if a_list:
                    all_actions.append(torch.tensor(a_list, dtype=torch.float32))
                if q_list:
                    all_qpos.append(torch.tensor(q_list, dtype=torch.float32))

    if not all_actions or not all_qpos:
        raise RuntimeError(
            f"No valid actions/qpos to compute stats under roots={roots}"
        )

    all_actions_t = torch.cat(all_actions, dim=0)
    all_qpos_t = torch.cat(all_qpos, dim=0)
    action_mean = all_actions_t.mean(dim=0).float().numpy()
    action_std = torch.clip(all_actions_t.std(dim=0), 1e-2).float().numpy()
    qpos_mean = all_qpos_t.mean(dim=0).float().numpy()
    qpos_std = torch.clip(all_qpos_t.std(dim=0), 1e-2).float().numpy()
    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
    }


def forward_pass(data, policy: ACTPolicy, device: torch.device, encode_command: bool = False):
    if len(data) == 5:
        image_data, qpos_data, action_data, is_pad, command_embedding = data
        command_embedding = command_embedding.to(device)
    else:
        image_data, qpos_data, action_data, is_pad = data
        command_embedding = None
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad, command_embedding, encode_command=encode_command)


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
    device = config.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    encode_command = config.get("encode_command", False)

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
            # Always use a single rolling latest file for simplicity.
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

    # print("---------------------------")
    # print(f"Shared backbone: {policy.model.shared_backbone}")
    # print("---------------------------")

    # import sys
    # sys.exit()


    optimizer = make_optimizer(policy)
    constant_lr = config.get("constant_lr", False)

    total_samples = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size
    batches_per_epoch = len(train_dataloader)
    
    # Scheduler steps once per epoch
    scheduler = None if constant_lr else make_scheduler(optimizer, num_epochs)
    
    # Mixed precision training (AMP)
    use_amp = config.get("use_amp", False)
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")

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
        sched_state = checkpoint.get("scheduler_state_dict")
        if scheduler is not None and sched_state is not None:
            scheduler.load_state_dict(sched_state)
        start_epoch = checkpoint["epoch"] + 1
        print(loading_status)
    else:
        start_epoch = 0

    policy.to(device)
    
    # Print model parameter count
    total_params, trainable_params, non_trainable_params = count_parameters(policy)
    param_info = (
        f"Model Parameters:\n"
        f"  Total: {total_params:,} ({format_number(total_params)})\n"
        f"  Trainable: {trainable_params:,} ({format_number(trainable_params)})\n"
        f"  Non-trainable: {non_trainable_params:,} ({format_number(non_trainable_params)})"
    )
    if logger:
        logger.info(param_info)
    else:
        print(param_info)
    
    # Print optimizer parameter groups info
    optimizer_info_parts = ["Optimizer Parameter Groups:"]
    for i, param_group in enumerate(optimizer.param_groups):
        group_params = sum(p.numel() for p in param_group['params'])
        lr = param_group.get('lr', 'N/A')
        weight_decay = param_group.get('weight_decay', 'N/A')
        optimizer_info_parts.append(
            f"  Group {i}: {group_params:,} params ({format_number(group_params)}), "
            f"lr={lr}, weight_decay={weight_decay}"
        )
    optimizer_info = "\n".join(optimizer_info_parts)
    if logger:
        logger.info(optimizer_info)
    else:
        print(optimizer_info)
    
    best_model_metric = config.get("best_model_metric", "total_loss")  # "total_loss" or "l1_loss"
    best_val = float("inf")
    best_ckpt_path = None
    
    # Restore best_val from checkpoint if available
    # Try to restore the value that matches the current metric
    if load_ckpt == "y" and checkpoint is not None:
        if best_model_metric == "l1_loss":
            # If using l1_loss metric, prefer best_val_l1 if available
            # (checkpoint saved with total_loss metric may have best_val_l1 as reference)
            if "best_val_l1" in checkpoint:
                best_val = checkpoint["best_val_l1"]
                print(f"Restored best_val (L1): {best_val:.5f}")
            elif "best_val" in checkpoint:
                # If best_val_l1 not available, use best_val (which might be l1 if saved with l1_loss metric)
                best_val = checkpoint["best_val"]
                if "best_val_total_loss" in checkpoint:
                    print(f"Restored best_val (L1): {best_val:.5f} (Total: {checkpoint['best_val_total_loss']:.5f})")
                else:
                    print(f"Restored best_val (L1): {best_val:.5f}")
        else:
            # Using total_loss metric, use best_val directly
            if "best_val" in checkpoint:
                best_val = checkpoint["best_val"]
                if "best_val_l1" in checkpoint:
                    print(f"Restored best_val (Total): {best_val:.5f} (L1: {checkpoint['best_val_l1']:.5f})")
                else:
                    print(f"Restored best_val (Total): {best_val:.5f}")
    
    if logger:
        logger.info(f"Best model selection metric: {best_model_metric}")
    else:
        print(f"Best model selection metric: {best_model_metric}")
    
    # Interval-based checkpoint tracking
    save_ckpt_every = config.get("save_ckpt_every", 100)
    disable_latest_checkpoint = config.get("disable_latest_checkpoint", False)
    save_latest_every = config.get("save_latest_every", 1)  # in epochs; 1 = every epoch
    interval_best_val_l1 = float("inf")
    interval_best_epoch = -1
    current_interval_start = start_epoch
    # Store best model state in GPU memory (will be written at interval end)
    interval_best_state = None
    interval_best_interval_idx = -1
    # Store global best model state in GPU memory (will be written at training end)
    global_best_state = None
    
    train_history: List[Dict[str, torch.Tensor]] = []

    epoch_bar = trange(start_epoch, num_epochs, desc="Epochs", leave=True)
    for epoch in epoch_bar:
        policy.train()
        optimizer.zero_grad()
        batch_bar = tqdm(train_dataloader, desc=f"Train {epoch+1}", leave=False)
        for batch_idx, data in enumerate(batch_bar):
            # Forward pass with optional AMP
            if use_amp:
                with autocast():
                    forward_dict = forward_pass(data, policy, device, encode_command)
                    loss = forward_dict["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                forward_dict = forward_pass(data, policy, device, encode_command)
                loss = forward_dict["loss"]
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            
            # Record epoch in training history for traceability
            history_entry = detach_dict(forward_dict)
            history_entry["epoch"] = torch.tensor(epoch, dtype=torch.long)
            train_history.append(history_entry)
            
            # Update tqdm postfix with loss, l1, and current learning rate
            postfix_dict = {"loss": f"{loss.item():.4f}"}
            if "l1" in forward_dict:
                postfix_dict["l1"] = f"{forward_dict['l1'].item():.4f}"
            if scheduler is not None:
                postfix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.5e}"
            batch_bar.set_postfix(postfix_dict)
        
        # Step scheduler at end of each epoch
        if scheduler is not None:
            scheduler.step()
        
        # Compute epoch training summary
        e = epoch - start_epoch
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * e : (batch_idx + 1) * (e + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
        
        # Log training summary
        train_msg = f"Epoch {epoch+1}: train_loss={epoch_train_loss:.5f}"
        if "l1" in epoch_summary:
            train_msg += f", train_l1={epoch_summary['l1']:.5f}"
        if current_lr is not None:
            train_msg += f", lr={current_lr:.5e}"
        if logger:
            logger.info(train_msg)
        else:
            tqdm.write(train_msg)

        # Validation at end of epoch
        val_loss = None
        val_l1 = None
        val_l1_infer = None  # Inference mode L1 (真实推理能力指标)
        if val_dataloader is not None:
            policy.eval()
            val_losses = []
            val_l1_losses = []
            val_l1_infer_losses = []  # 推理模式 L1 losses
            num_queries = policy.num_queries
            with torch.inference_mode():
                val_bar = tqdm(val_dataloader, desc=f"Val {epoch}", leave=False)
                for data in val_bar:
                    # 解包数据
                    if len(data) == 5:
                        image_data, qpos_data, action_data, is_pad, command_embedding = data
                        command_embedding = command_embedding.to(device)
                    else:
                        image_data, qpos_data, action_data, is_pad = data
                        command_embedding = None
                    image_data = image_data.to(device)
                    qpos_data = qpos_data.to(device)
                    action_data = action_data.to(device)
                    is_pad = is_pad.to(device)
                    
                    # 1. 重建模式 Loss（传入 GT actions，用于监控 VAE 训练）
                    if use_amp:
                        with autocast():
                            forward_dict = policy(qpos_data, image_data, action_data, is_pad, command_embedding, encode_command=encode_command)
                    else:
                        forward_dict = policy(qpos_data, image_data, action_data, is_pad, command_embedding, encode_command=encode_command)
                    val_losses.append(forward_dict["loss"].item())
                    if "l1" in forward_dict:
                        val_l1_losses.append(forward_dict["l1"].item())
                    
                    # 2. 推理模式 Loss（不传入 actions，使用零向量 latent，反映真实推理能力）
                    if use_amp:
                        with autocast():
                            a_hat_infer = policy(qpos_data, image_data, actions=None, is_pad=None, command_embedding=command_embedding, encode_command=encode_command)
                    else:
                        a_hat_infer = policy(qpos_data, image_data, actions=None, is_pad=None, command_embedding=command_embedding, encode_command=encode_command)
                    
                    # 计算推理模式 L1 loss（手动计算，排除 padding）
                    action_gt = action_data[:, :num_queries]
                    is_pad_slice = is_pad[:, :num_queries]
                    mask = (~is_pad_slice).unsqueeze(-1)  # [batch, seq, 1]
                    action_dim = action_gt.shape[-1]
                    valid = mask.sum() * action_dim
                    l1_infer = (F.l1_loss(action_gt, a_hat_infer, reduction="none") * mask).sum() / valid.clamp(min=1)
                    val_l1_infer_losses.append(l1_infer.item())
                    
                    val_bar.set_postfix({
                        "loss": f"{val_losses[-1]:.4f}",
                        "l1_infer": f"{val_l1_infer_losses[-1]:.4f}"
                    })
            
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            val_l1 = float(np.mean(val_l1_losses)) if val_l1_losses else None
            val_l1_infer = float(np.mean(val_l1_infer_losses)) if val_l1_infer_losses else None
            
            # Log validation metrics
            val_msg = f"Epoch {epoch+1}: val_loss={val_loss:.5f}"
            if val_l1 is not None:
                val_msg += f", val_l1={val_l1:.5f}"
            if val_l1_infer is not None:
                val_msg += f", val_l1_infer={val_l1_infer:.5f}"
                # 显示重建 vs 推理的 gap（gap 越大说明模型越依赖 encoder 提供的 GT 信息）
                if val_l1 is not None:
                    gap = val_l1_infer - val_l1
                    val_msg += f" (gap={gap:+.5f})"
            if logger:
                logger.info(val_msg)
            else:
                tqdm.write(val_msg)
            
            # Update global best model based on validation metric
            if best_model_metric == "l1_loss_infer":
                current_val_metric = val_l1_infer if val_l1_infer is not None else val_l1
            elif best_model_metric == "l1_loss":
                current_val_metric = val_l1 if val_l1 is not None else val_loss
            else:
                current_val_metric = val_loss
            
            if current_val_metric is not None and current_val_metric < best_val:
                best_val = current_val_metric
                global_best_state = {
                    "model_state_dict": policy.serialize(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "val_loss": val_loss,
                    "val_l1": val_l1,
                    "val_l1_infer": val_l1_infer,  # 推理模式 L1
                }
                infer_str = f", val_l1_infer={val_l1_infer:.5f}" if val_l1_infer is not None else ""
                if logger:
                    logger.info(f"New best model at epoch {epoch+1} with {best_model_metric}={current_val_metric:.5f}{infer_str}")
                else:
                    tqdm.write(f"New best model at epoch {epoch+1} with {best_model_metric}={current_val_metric:.5f}{infer_str}")

        # Save latest checkpoint every N epochs (rolling backup) - optional
        # This allows resuming from recent epochs, not just multiples of save_ckpt_every
        # Controlled by:
        #   - disable_latest_checkpoint: completely disable latest checkpoint saving
        #   - save_latest_every: save every N epochs (default: 1 = every epoch)
        if (not disable_latest_checkpoint) and save_latest_every > 0:
            # Use relative epoch index so behavior is consistent when resuming,
            # but always overwrite the same latest checkpoint file (rolling update).
            rel_epoch_idx = epoch - start_epoch + 1  # 1-based within this run
            if rel_epoch_idx % save_latest_every == 0:
                latest_ckpt_path = os.path.join(ckpt_dir, f"policy_latest_seed_{seed}.ckpt")
                legacy_latest_path = os.path.join(ckpt_dir, "policy_latest.ckpt")
                latest_payload = {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                }
                if scheduler is not None:
                    latest_payload["scheduler_state_dict"] = scheduler.state_dict()

                if os.path.exists(legacy_latest_path):
                    try:
                        os.remove(legacy_latest_path)
                    except Exception:
                        pass
                torch.save(latest_payload, latest_ckpt_path)

        # Interval-based best checkpoint saving: store best model in GPU memory,
        # write to disk only at interval end to reduce IO pressure
        # Check if we've entered a new interval
        current_interval_idx = (epoch - start_epoch) // save_ckpt_every
        interval_start_epoch = start_epoch + current_interval_idx * save_ckpt_every
        
        if interval_start_epoch > current_interval_start:
            # We've entered a new interval, write previous interval's best checkpoint to disk
            if interval_best_state is not None and interval_best_epoch >= 0:
                prev_interval_start = start_epoch + interval_best_interval_idx * save_ckpt_every
                prev_interval_end = start_epoch + (interval_best_interval_idx + 1) * save_ckpt_every
                # Format metric value for filename based on best_model_metric (replace '.' with 'p' to avoid path issues)
                if best_model_metric == "l1_loss_infer":
                    metric_val = interval_best_state.get("val_l1_infer")
                    metric_fname = f"_l1infer_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
                elif best_model_metric == "l1_loss":
                    metric_val = interval_best_state.get("val_l1")
                    metric_fname = f"_l1_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
                else:
                    metric_val = interval_best_state.get("val_loss")
                    metric_fname = f"_loss_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
                prev_ckpt_path = os.path.join(
                    ckpt_dir, f"policy_interval_{interval_best_interval_idx}_epoch_{interval_best_epoch}{metric_fname}_best_seed_{seed}.ckpt"
                )
                # Prepare payload for saving: for interval-best checkpoints we only
                # keep model weights and validation metrics to keep files compact.
                save_payload = {
                    "model_state_dict": interval_best_state["model_state_dict"],
                    "epoch": interval_best_state["epoch"],
                    "val_l1": interval_best_state.get("val_l1"),
                    "val_loss": interval_best_state.get("val_loss"),
                    "val_l1_infer": interval_best_state.get("val_l1_infer"),
                }
                torch.save(save_payload, prev_ckpt_path)
                val_l1_str = f"{interval_best_state.get('val_l1', 'N/A'):.5f}" if interval_best_state.get("val_l1") is not None else "N/A"
                val_l1_infer_str = f", val_l1_infer: {interval_best_state.get('val_l1_infer'):.5f}" if interval_best_state.get("val_l1_infer") is not None else ""
                if logger:
                    logger.info(
                        f"Saved interval [{prev_interval_start}-{prev_interval_end}) "
                        f"best checkpoint (epoch {interval_best_epoch}, val_l1: {val_l1_str}{val_l1_infer_str}, val_loss: {interval_best_state.get('val_loss', 'N/A'):.5f}) "
                        f"to {prev_ckpt_path}"
                    )
                else:
                    tqdm.write(
                        f"Saved interval [{prev_interval_start}-{prev_interval_end}) "
                        f"best checkpoint (epoch {interval_best_epoch}, val_l1: {val_l1_str}{val_l1_infer_str}, val_loss: {interval_best_state.get('val_loss', 'N/A'):.5f})"
                    )
            
            # Reset for new interval
            interval_best_val_l1 = float("inf")
            interval_best_epoch = -1
            interval_best_state = None
            interval_best_interval_idx = -1
            current_interval_start = interval_start_epoch
        
        # Update best checkpoint in current interval if validation is available
        # Store state in GPU memory, will be written at interval end
        if val_dataloader is not None:
            # 按 best_model_metric 选择验证指标
            if best_model_metric == "l1_loss_infer":
                current_val_metric = val_l1_infer if val_l1_infer is not None else val_l1
            elif best_model_metric == "l1_loss":
                current_val_metric = val_l1 if val_l1 is not None else val_loss
            else:
                current_val_metric = val_loss if val_loss is not None else val_l1
            
            if current_val_metric is not None and current_val_metric < interval_best_val_l1:
                interval_best_val_l1 = current_val_metric
                interval_best_epoch = epoch
                interval_best_interval_idx = current_interval_idx
                # Store best model state in GPU memory
                interval_best_state = {
                    "model_state_dict": policy.serialize(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "val_l1": val_l1 if val_l1 is not None else None,
                    "val_loss": val_loss,
                    "val_l1_infer": val_l1_infer,  # 推理模式 L1
                }
                val_l1_str = f"{val_l1:.5f}" if val_l1 is not None else "N/A"
                val_l1_infer_str = f", val_l1_infer: {val_l1_infer:.5f}" if val_l1_infer is not None else ""
                if logger:
                    logger.info(
                        f"New interval [{start_epoch + current_interval_idx * save_ckpt_every}-{start_epoch + (current_interval_idx + 1) * save_ckpt_every}) "
                        f"best model (epoch {epoch}, val_l1: {val_l1_str}{val_l1_infer_str}, val_loss: {val_loss:.5f}) - stored in GPU memory"
                    )
                else:
                    tqdm.write(
                        f"New interval [{start_epoch + current_interval_idx * save_ckpt_every}-{start_epoch + (current_interval_idx + 1) * save_ckpt_every}) "
                        f"best model (epoch {epoch}, val_l1: {val_l1_str}{val_l1_infer_str}, val_loss: {val_loss:.5f}) - stored in GPU memory"
                    )

    # Write the last interval's best checkpoint if it exists
    if interval_best_state is not None and interval_best_epoch >= 0:
        last_interval_start = start_epoch + interval_best_interval_idx * save_ckpt_every
        last_interval_end = start_epoch + (interval_best_interval_idx + 1) * save_ckpt_every
        # Format metric value for filename based on best_model_metric
        if best_model_metric == "l1_loss_infer":
            metric_val = interval_best_state.get("val_l1_infer")
            metric_fname = f"_l1infer_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
        elif best_model_metric == "l1_loss":
            metric_val = interval_best_state.get("val_l1")
            metric_fname = f"_l1_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
        else:
            metric_val = interval_best_state.get("val_loss")
            metric_fname = f"_loss_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
        last_ckpt_path = os.path.join(
            ckpt_dir, f"policy_interval_{interval_best_interval_idx}_epoch_{interval_best_epoch}{metric_fname}_best_seed_{seed}.ckpt"
        )
        save_payload = {
            "model_state_dict": interval_best_state["model_state_dict"],
            "epoch": interval_best_state["epoch"],
            "val_l1": interval_best_state.get("val_l1"),
            "val_loss": interval_best_state.get("val_loss"),
            "val_l1_infer": interval_best_state.get("val_l1_infer"),
        }
        torch.save(save_payload, last_ckpt_path)
        val_l1_str = f"{interval_best_state.get('val_l1', 'N/A'):.5f}" if interval_best_state.get("val_l1") is not None else "N/A"
        val_l1_infer_str = f", val_l1_infer: {interval_best_state.get('val_l1_infer'):.5f}" if interval_best_state.get("val_l1_infer") is not None else ""
        if logger:
            logger.info(
                f"Saved final interval [{last_interval_start}-{last_interval_end}) "
                f"best checkpoint (epoch {interval_best_epoch}, val_l1: {val_l1_str}{val_l1_infer_str}, val_loss: {interval_best_state.get('val_loss', 'N/A'):.5f}) "
                f"to {last_ckpt_path}"
            )
        else:
            tqdm.write(
                f"Saved final interval [{last_interval_start}-{last_interval_end}) "
                f"best checkpoint (epoch {interval_best_epoch}, val_l1: {val_l1_str}{val_l1_infer_str}, val_loss: {interval_best_state.get('val_loss', 'N/A'):.5f})"
            )
    
    # Write global best model checkpoint if it exists
    if global_best_state is not None:
        # Format metric value for filename based on best_model_metric
        if best_model_metric == "l1_loss_infer":
            metric_val = global_best_state.get("val_l1_infer")
            metric_fname = f"_l1infer_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
        elif best_model_metric == "l1_loss":
            metric_val = global_best_state.get("val_l1")
            metric_fname = f"_l1_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
        else:
            metric_val = global_best_state.get("val_loss")
            metric_fname = f"_loss_{metric_val:.5f}".replace(".", "p") if metric_val is not None else ""
        best_ckpt_path = os.path.join(ckpt_dir, f"policy_best{metric_fname}_seed_{seed}.ckpt")
        save_payload = {
            "model_state_dict": global_best_state["model_state_dict"],
            "epoch": global_best_state["epoch"],
            "best_val": global_best_state["best_val"],
            "val_loss": global_best_state.get("val_loss"),
            "val_l1": global_best_state.get("val_l1"),
            "val_l1_infer": global_best_state.get("val_l1_infer"),
        }
        torch.save(save_payload, best_ckpt_path)
        val_l1_str = f"{global_best_state.get('val_l1'):.5f}" if global_best_state.get("val_l1") is not None else "N/A"
        val_loss_str = f"{global_best_state.get('val_loss'):.5f}" if global_best_state.get("val_loss") is not None else "N/A"
        val_l1_infer_str = f", val_l1_infer={global_best_state.get('val_l1_infer'):.5f}" if global_best_state.get("val_l1_infer") is not None else ""
        msg = f"Saved global best checkpoint (epoch {global_best_state['epoch']}, val_loss={val_loss_str}, val_l1={val_l1_str}{val_l1_infer_str}) to {best_ckpt_path}"
        if logger:
            logger.info(msg)
        else:
            tqdm.write(msg)
    # Save final checkpoint (last epoch state)
    ckpt_path_final = os.path.join(ckpt_dir, "policy_last.ckpt")
    final_payload = {
        "model_state_dict": policy.serialize(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler is not None:
        final_payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(final_payload, ckpt_path_final)
    if logger:
        logger.info(f"Saved final checkpoint to {ckpt_path_final}")

    legacy_latest_path = os.path.join(ckpt_dir, f"policy_latest_seed_{seed}.ckpt")
    if os.path.exists(legacy_latest_path):
        try:
            os.remove(legacy_latest_path)
            if logger:
                logger.info(f"Removed legacy latest checkpoint {legacy_latest_path}")
        except Exception:
            pass


def main(args: Dict):
    set_seed(args["seed"])

    # Setup device
    gpu_id = args.get("gpu", None)
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA is not available, but --gpu {gpu_id} was specified")
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU ID {gpu_id}. Available GPUs: 0-{torch.cuda.device_count() - 1}")
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # Log device information
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU {device.index}: {torch.cuda.get_device_name(device.index)}")

    use_language = args["use_language"]
    language_encoder = args["language_encoder"]
    use_state = not args.get("no_state", False)

    dataset_dirs: List[str] = []
    num_episodes_list: List[int] = []
    camera_names: Sequence[str] = []
    max_episode_len = 0

    use_splitted = args.get("use_splitted", False)
    splitted_root = args.get("splitted_root")
    stage_embeddings_file = args.get("stage_embeddings_file")

    # Simple HDF5 episodic dataset interface (EpisodicDataset / SRTDataset in llp.utils)
    dataset_dir = args.get("dataset_dir")
    num_episodes = args.get("num_episodes")
    hdf5_use_srt = args.get("hdf5_use_srt", False)
    val_batch_size = args.get("val_batch_size") or args["batch_size"]
    train_ratio = args.get("train_ratio", 0.8)

    if dataset_dir is not None:
        # Auto-detect num_episodes if not provided
        if num_episodes is None:
            # Find all episode_*.hdf5 files and get the maximum episode ID
            episode_files = glob.glob(os.path.join(dataset_dir, "episode_*.hdf5"))
            if not episode_files:
                raise ValueError(f"No episode_*.hdf5 files found in {dataset_dir}")
            # Extract episode IDs and find max
            episode_ids = []
            for f in episode_files:
                try:
                    # Extract number from "episode_123.hdf5"
                    basename = os.path.basename(f)
                    ep_id = int(basename.split("_")[1].split(".")[0])
                    episode_ids.append(ep_id)
                except (ValueError, IndexError):
                    continue
            if not episode_ids:
                raise ValueError(f"Could not parse episode IDs from files in {dataset_dir}")
            # num_episodes should be max_id + 1 (assuming IDs start from 0)
            num_episodes = max(episode_ids) + 1
            if logger:
                logger.info(f"Auto-detected num_episodes={num_episodes} from {dataset_dir}")
            else:
                print(f"Auto-detected num_episodes={num_episodes} from {dataset_dir}")
        
        # Directly load from a directory of episode_*.hdf5 files
        camera_names = args.get("camera_names", ["left_frame", "right_frame"])
        train_dataloader, val_dataloader, stats, _ = load_data(
            dataset_dir=dataset_dir,
            num_episodes=num_episodes,
            camera_names=camera_names,
            batch_size_train=args["batch_size"],
            batch_size_val=val_batch_size,
            train_ratio=train_ratio,
            prefetch_factor=args.get("prefetch_factor", 0),
            num_workers=args.get("num_workers", 0),
        )
        # For simple episodic datasets we don't use explicit max_episode_len here;
        # chunk_size still controls the number of queries.
        max_episode_len = args.get("chunk_size", 30)

    elif use_splitted:
        if not splitted_root:
            raise ValueError("--splitted_root must be provided when --use_splitted is set")
        camera_names = args.get("camera_names", ["left_frame", "right_frame"])
        max_episode_len = args.get("chunk_size", 30)
        # 只用训练集计算统计量，避免数据泄露
        val_root = args.get("val_splitted_root")
        stats = compute_splitted_norm_stats([splitted_root])
        train_dataloader, stats, _ = load_splitted_data(
            root_dir=splitted_root,
            camera_names=camera_names,
            batch_size_train=args["batch_size"],
            max_len=max_episode_len,
            norm_stats=stats,
            use_language=use_language,
            stage_embeddings_file=stage_embeddings_file,
            use_augmentation=args.get("use_augmentation", False),
            image_size=args.get("image_size", 224),
            use_episodic_sampling=args.get("use_episodic_sampling", False),
            use_state=use_state,
            prefetch_factor=args.get("prefetch_factor", 2),
            num_workers=args.get("num_workers", 8),
        )
        val_dataloader = None
        if val_root:
            stage_embeddings = None
            if use_language and stage_embeddings_file and os.path.exists(stage_embeddings_file):
                with open(stage_embeddings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("stage_embeddings", data)
                # Convert to 0-indexed to match stage_idx in SplittedEpisodicDataset
                stage_embeddings = {
                    int(e["stage"]) - 1: e["embedding"]
                    for e in entries
                    if "stage" in e and "embedding" in e
                }
            val_dataset = SplittedEpisodicDataset(
                val_root,
                camera_names,
                stats,  # 与 train 共享同一统计量
                max_len=max_episode_len,
                use_language=use_language,
                stage_embeddings=stage_embeddings,
                image_size=args.get("image_size", 224),
                use_augmentation=False,  # No augmentation for validation
                training=False,
                use_episodic_sampling=args.get("use_episodic_sampling", False),
                use_state=use_state,
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
            use_state=use_state,
        )
        val_dataloader = None

    stats["use_state"] = use_state
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    # Determine shared_backbone: use explicit arg if provided, otherwise compute default
    if "shared_backbone" in args and args["shared_backbone"] is not None:
        shared_backbone = args["shared_backbone"]
    else:
        # Default logic: share backbone if not using language and not using film
        shared_backbone = not use_language and "film" not in args["image_encoder"]
    
    policy_config = {
        "lr": args["lr"],
        "num_queries": args["chunk_size"],
        "kl_weight": args["kl_weight"],
        "hidden_dim": args["hidden_dim"],
        "dim_feedforward": args["dim_feedforward"],
        "lr_backbone": args.get("lr_backbone", 1e-5),
        "weight_decay": args.get("weight_decay", 1e-4),
        "backbone": args["image_encoder"],
        "enc_layers": args.get("enc_layers_num", 4),
        "dec_layers": args.get("dec_layers_num", 7),
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
        "use_state": use_state,
        "shared_backbone": shared_backbone,
        "use_gated_attention": args.get("use_gated_attention", False),
        "mlp_type": args.get("mlp_type", "swiglu"),
        "gate_mode": args.get("gate_mode", "element-wise"),
    }

    config = {
        "num_epochs": args["num_epochs"],
        "ckpt_dir": ckpt_dir,
        "seed": args["seed"],
        "policy_config": policy_config,
        "log_every": args.get("log_every", 1),
        "best_model_metric": args.get("best_model_metric", "total_loss"),
        "constant_lr": args.get("constant_lr", False),
        "save_ckpt_every": args.get("save_ckpt_every", 20),
        "disable_latest_checkpoint": args.get("disable_latest_checkpoint", False),
        "save_latest_every": args.get("save_latest_every", 1),
        "device": device,
    }

    train_bc(train_dataloader, config, val_dataloader=val_dataloader, logger=logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training (AMP) to reduce GPU memory usage")

    # Model options
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--enc_layers_num", type=int, default=4)
    parser.add_argument("--dec_layers_num", type=int, default=7)
    parser.add_argument("--image_encoder", type=str, default="efficientnet_b3film")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (default: auto-select first available GPU, or CPU if no GPU available)")
    parser.add_argument("--mlp_type", type=str, default="swiglu", choices=["swiglu", "standard"], help="MLP block type for Transformer FFN")
    parser.add_argument("--gate_mode", type=str, default="element-wise", choices=["element-wise", "head-wise"], help="Gated attention mode")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--no_state", action="store_true", help="Disable qpos/state inputs throughout training/inference pipeline")
    parser.add_argument("--no_encoder", action="store_true", help="Disable VAE encoder, use zero latent vector instead")
    parser.add_argument("--vq", action="store_true", help="Use Vector Quantization instead of standard VAE")
    parser.add_argument("--vq_class", type=int, default=512, help="Number of VQ codebook classes")
    parser.add_argument("--vq_dim", type=int, default=32, help="Dimension of each VQ codebook entry")
    parser.add_argument(
        "--use_gated_attention",
        action="store_true",
        help="Use gated attention in DETR transformer and encoder (ablation flag)",
    )
    parser.add_argument("--shared_backbone", type=lambda x: x.lower() in ('true', '1', 'yes'), default=None, help="Whether to share backbone across cameras (default: auto-detect based on use_language and image_encoder)")

    parser.add_argument("--use_language", action="store_true")
    parser.add_argument("--language_encoder", type=str, default="distilbert", choices=["distilbert", "clip"])
    parser.add_argument("--command_list", nargs="*", default=[])
    parser.add_argument("--encode_command", action="store_true", help="Enable encoding command_embedding in CVAE encoder (default: disabled, decoder will still use it if use_language=True)")

    # Dataset options
    parser.add_argument("--use_splitted", action="store_true")
    parser.add_argument("--splitted_root", type=str)
    parser.add_argument("--val_splitted_root", type=str)
    parser.add_argument("--stage_embeddings_file", type=str)
    parser.add_argument("--camera_names", nargs="*", default=["left_frame", "right_frame"])
    parser.add_argument("--use_augmentation", action="store_true", help="Enable data augmentation for images")
    parser.add_argument("--image_size", type=int, default=224, help="Image size for augmentation")
    parser.add_argument("--use_episodic_sampling", action="store_true", help="Use episodic sampling mode (similar to EpisodicDataset): traverse run IDs, randomly sample stage, then start_ts. Default: single random mode (traverse all stage/run, randomly sample start_ts)")
    
    # DataLoader options
    parser.add_argument("--prefetch_factor", type=int, default=0, help="Prefetch factor for the dataset")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for the dataset")


    parser.add_argument("--task_config", type=json.loads, default=None)
    parser.add_argument("--log_every", type=int, default=1)
    
    # Image encoder training control
    parser.add_argument("--train_image_encoder", action="store_true", help="Enable training of the image encoder (default: frozen)")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for image encoder backbone (only used if --train_image_encoder is set)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    
    # Best model selection metric
    parser.add_argument(
        "--best_model_metric",
        type=str,
        default="l1_loss_infer",
        choices=["total_loss", "l1_loss", "l1_loss_infer"],
        help="Metric for selecting best model: 'total_loss' (l1 + kl_weight * kl), 'l1_loss' (reconstruction L1), or 'l1_loss_infer' (default, inference-mode L1 - reflects true inference capability)"
    )
    parser.add_argument("--constant_lr", action="store_true", help="Disable learning rate scheduler and keep LR constant")
    parser.add_argument("--save_ckpt_every", type=int, default=20, help="Interval size for saving best checkpoint within each interval [i*n, (i+1)*n) based on val l1 loss (default: 100)")
    parser.add_argument("--disable_latest_checkpoint", action="store_true", help="Disable saving latest checkpoint every epoch to reduce disk IO (interval and global best models are still saved)")
    parser.add_argument(
        "--save_latest_every",
        type=int,
        default=1,
        help="Save latest checkpoint every N epochs (default: 1). Ignored if --disable_latest_checkpoint is set.",
    )

    # Simple HDF5 episodic dataset options (EpisodicDataset / SRTDataset in llp.utils)
    parser.add_argument("--dataset_dir", type=str, help="Directory containing episode_*.hdf5 files")
    parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes in dataset_dir (default: auto-detect from max episode ID)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data (default: 0.8)")
    parser.add_argument("--val_batch_size", type=int, help="Validation batch size (default: same as --batch_size)")
    parser.add_argument("--hdf5_use_srt", action="store_true", help="Use SRTDataset with Albumentations instead of plain EpisodicDataset")

    args = parser.parse_args()
    main(vars(args))
