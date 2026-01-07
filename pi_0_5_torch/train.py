"""
Pi0.5 Training Script - Fully aligned with OpenPI official implementation.

This training script handles:
1. Dataset loading with Pi05Dataset (standard) or Pi05HierarchicalDataset (hierarchical)
2. Preprocessing with Pi05Preprocessor
3. Training modes:
   - Action-only: Standard flow matching training
   - Hierarchical: Joint subtask prediction + action generation (Pi0.5 paper Eq. 1)
4. Gradient clipping and proper optimizer configuration
5. Learning rate scheduling (optional)

Training Loss (Pi0.5 Paper Equation 1):
    L = H(subtask_ids, logits) + α * ||u_t - v_t||²
    
where:
    - H: Cross-entropy loss for subtask text prediction
    - Second term: Flow matching loss for action generation
    - α: Loss weight (default 10.0 from paper)
"""

import argparse
import logging
import os
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from .configuration import PI05Config
from .dataset import Pi05Dataset, Pi05HierarchicalDataset, pi05_collate_fn
from .model import PI05Model
from .preprocess import Pi05DatasetStats, Pi05Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataloaders(
    data_root: str,
    primary_camera: str,
    instruction_json: Optional[str],
    config: PI05Config,
    batch_size: int,
    num_workers: int = 8,
    validation_split: float = 0.0,
    enable_hierarchical: bool = False,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optionally validation) dataloaders.
    
    Args:
        data_root: Root directory with stage*/run_* structure
        primary_camera: Camera folder name
        instruction_json: JSON with stage->text mapping
        config: Pi0.5 config
        batch_size: Batch size
        num_workers: Number of dataloader workers
        validation_split: Fraction for validation
        enable_hierarchical: Enable hierarchical dataset with high_level_task/subtask
    """
    if enable_hierarchical:
        dataset = Pi05HierarchicalDataset(
            root_dir=data_root,
            primary_camera=primary_camera,
            chunk_size=config.chunk_size,
            instruction_json=instruction_json,
            use_state=True,
            training_mode="joint",
        )
    else:
        dataset = Pi05Dataset(
            root_dir=data_root,
            primary_camera=primary_camera,
            chunk_size=config.chunk_size,
            instruction_json=instruction_json,
            use_state=True,
            enable_hierarchical=enable_hierarchical,
        )

    val_loader = None
    if validation_split > 0 and len(dataset) > 10:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=pi05_collate_fn,
        )
        dataset = train_dataset

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pi05_collate_fn,
    )

    return train_loader, val_loader


def build_model_and_preprocessor(
    config: PI05Config,
    device: str,
    dataset_stats_path: Optional[str] = None,
    use_augmentation: bool = True,
    fast_tokenizer_path: Optional[str] = None,
) -> tuple[PI05Model, Pi05Preprocessor]:
    """Build model and preprocessor with optional dataset statistics."""
    # Load dataset statistics if provided
    stats = Pi05DatasetStats()
    if dataset_stats_path is not None and os.path.exists(dataset_stats_path):
        data = torch.load(dataset_stats_path, map_location="cpu")
        # Expected format:
        # {
        #   "state": {"mean": Tensor[max_state_dim], "std": Tensor[max_state_dim]},
        #   "action": {"mean": Tensor[act_dim], "std": Tensor[act_dim]},
        # }
        stats.state_stats = data.get("state")
        stats.action_stats = data.get("action")
        logger.info(f"Loaded dataset statistics from {dataset_stats_path}")

    preprocessor = Pi05Preprocessor(
        config=config,
        dataset_stats=stats,
        device=device,
        use_augmentation=use_augmentation,
        fast_tokenizer_path=fast_tokenizer_path,
    )

    model = PI05Model(config)
    model.to(device)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model, preprocessor


def train_one_epoch(
    model: PI05Model,
    preprocessor: Pi05Preprocessor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    gradient_clip_norm: float = 1.0,
    log_interval: int = 50,
) -> float:
    """
    Train for one epoch.

    Uses loss with reduction="none" and performs mean aggregation externally,
    matching OpenPI's approach.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader):
        # Preprocess batch
        inputs = preprocessor.preprocess_batch(batch, train=True)

        # Forward pass - returns loss with reduction="none"
        loss_per_elem = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            actions=inputs["actions"],
        )

        # Aggregate loss (mean over all dimensions)
        # You can customize this for weighted loss, masking padded actions, etc.
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad.any():
            # Mask out padded actions
            mask = ~action_is_pad.unsqueeze(-1).expand_as(loss_per_elem)
            loss = (loss_per_elem * mask).sum() / mask.sum()
        else:
            loss = loss_per_elem.mean()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        total_loss += float(loss.detach().cpu())
        num_steps += 1

        if step % log_interval == 0:
            logger.info(f"[Epoch {epoch}] Step {step}: loss = {loss.item():.6f}")

    avg_loss = total_loss / max(1, num_steps)
    logger.info(f"[Epoch {epoch}] Average loss: {avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def validate(
    model: PI05Model,
    preprocessor: Pi05Preprocessor,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Validate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_steps = 0

    for batch in dataloader:
        inputs = preprocessor.preprocess_batch(batch, train=False)

        loss_per_elem = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            actions=inputs["actions"],
        )

        action_is_pad = inputs["action_is_pad"]
        if action_is_pad.any():
            mask = ~action_is_pad.unsqueeze(-1).expand_as(loss_per_elem)
            loss = (loss_per_elem * mask).sum() / mask.sum()
        else:
            loss = loss_per_elem.mean()

        total_loss += float(loss.detach().cpu())
        num_steps += 1

    return total_loss / max(1, num_steps)


# ============================================================================
# Hierarchical Training (Pi0.5 Paper Equation 1)
# ============================================================================


def train_one_epoch_hierarchical(
    model: PI05Model,
    preprocessor: Pi05Preprocessor,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    alpha: float = 10.0,
    gradient_clip_norm: float = 1.0,
    log_interval: int = 50,
) -> dict[str, float]:
    """
    Train for one epoch using hierarchical Pi0.5 training (Equation 1).
    
    This jointly trains:
    1. Subtask prediction (cross-entropy loss)
    2. Action generation (flow matching loss)
    
    Combined loss:
        L = H(subtask_ids, logits) + α * ||u_t - v_t||²
    
    Args:
        model: Pi0.5 model
        preprocessor: Preprocessor instance
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device string
        epoch: Current epoch number
        alpha: Weight for flow matching loss (default 10.0 from paper)
        gradient_clip_norm: Gradient clipping norm
        log_interval: Logging interval
        
    Returns:
        Dictionary with loss components: {"total", "action", "text"}
    """
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_text_loss = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader):
        # Preprocess batch with hierarchical fields
        inputs = preprocessor.preprocess_hierarchical_batch(batch, train=True)

        # Forward pass with hierarchical output
        output = model.forward_hierarchical(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["high_level_input_ids"],
            attention_mask=inputs["high_level_attention_mask"],
            actions=inputs["actions"],
            subtask_ids=inputs["subtask_input_ids"],
            subtask_mask=inputs["subtask_attention_mask"],
            fast_action_ids=inputs.get("fast_action_ids"),
            fast_action_mask=inputs.get("fast_action_mask"),
            alpha=alpha,
        )

        # Get total loss
        loss = output.total_loss

        # Mask padded actions if needed
        action_is_pad = inputs["action_is_pad"]
        if action_is_pad.any() and output.action_loss is not None:
            mask = ~action_is_pad.unsqueeze(-1).expand_as(output.action_loss)
            action_loss_masked = (output.action_loss * mask).sum() / mask.sum()
        else:
            action_loss_masked = output.action_loss.mean() if output.action_loss is not None else 0.0

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        # Track losses
        total_loss += float(loss.detach().cpu())
        total_action_loss += float(action_loss_masked.detach().cpu()) if torch.is_tensor(action_loss_masked) else action_loss_masked
        if output.text_loss is not None:
            total_text_loss += float(output.text_loss.mean().detach().cpu())
        num_steps += 1

        if step % log_interval == 0:
            text_loss_val = output.text_loss.mean().item() if output.text_loss is not None else 0.0
            logger.info(
                f"[Epoch {epoch}] Step {step}: "
                f"total_loss = {loss.item():.6f}, "
                f"action_loss = {action_loss_masked.item() if torch.is_tensor(action_loss_masked) else action_loss_masked:.6f}, "
                f"text_loss = {text_loss_val:.6f}"
            )

    avg_total = total_loss / max(1, num_steps)
    avg_action = total_action_loss / max(1, num_steps)
    avg_text = total_text_loss / max(1, num_steps)
    
    logger.info(
        f"[Epoch {epoch}] Average losses: "
        f"total = {avg_total:.6f}, action = {avg_action:.6f}, text = {avg_text:.6f}"
    )
    
    return {
        "total": avg_total,
        "action": avg_action,
        "text": avg_text,
    }


@torch.no_grad()
def validate_hierarchical(
    model: PI05Model,
    preprocessor: Pi05Preprocessor,
    dataloader: DataLoader,
    device: str,
    alpha: float = 10.0,
) -> dict[str, float]:
    """Validate model with hierarchical loss."""
    model.eval()
    total_loss = 0.0
    total_action_loss = 0.0
    total_text_loss = 0.0
    num_steps = 0

    for batch in dataloader:
        inputs = preprocessor.preprocess_hierarchical_batch(batch, train=False)

        output = model.forward_hierarchical(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["high_level_input_ids"],
            attention_mask=inputs["high_level_attention_mask"],
            actions=inputs["actions"],
            subtask_ids=inputs["subtask_input_ids"],
            subtask_mask=inputs["subtask_attention_mask"],
            fast_action_ids=inputs.get("fast_action_ids"),
            fast_action_mask=inputs.get("fast_action_mask"),
            alpha=alpha,
        )

        action_is_pad = inputs["action_is_pad"]
        if action_is_pad.any() and output.action_loss is not None:
            mask = ~action_is_pad.unsqueeze(-1).expand_as(output.action_loss)
            action_loss = (output.action_loss * mask).sum() / mask.sum()
        else:
            action_loss = output.action_loss.mean() if output.action_loss is not None else 0.0

        total_loss += float(output.total_loss.detach().cpu())
        total_action_loss += float(action_loss.detach().cpu()) if torch.is_tensor(action_loss) else action_loss
        if output.text_loss is not None:
            total_text_loss += float(output.text_loss.mean().detach().cpu())
        num_steps += 1

    return {
        "total": total_loss / max(1, num_steps),
        "action": total_action_loss / max(1, num_steps),
        "text": total_text_loss / max(1, num_steps),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Pi0.5 policy on custom dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory with stage*/run_*")
    parser.add_argument("--primary_camera", type=str, default="left_frame", help="Camera folder name")
    parser.add_argument("--instruction_json", type=str, default=None, help="JSON with stage->text mapping")
    parser.add_argument("--dataset_stats", type=str, default=None, help="Path to .pt file with stats")
    parser.add_argument("--output_dir", type=str, default="./pi05_checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-5, help="Learning rate (matches OpenPI default)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--validation_split", type=float, default=0.05)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--use_augmentation", action="store_true", help="Use data augmentation")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    
    # Hierarchical training options (Pi0.5 paper)
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Enable hierarchical training (subtask prediction + action generation)"
    )
    parser.add_argument(
        "--alpha", type=float, default=10.0,
        help="Weight for action loss in hierarchical training (Eq. 1, default 10.0)"
    )
    parser.add_argument(
        "--planar_mode", action="store_true",
        help="High-level policy finetune (text CE only). Forces hierarchical dataloader; sets alpha=0."
    )
    parser.add_argument(
        "--use_fast_tokens", action="store_true",
        help="Use FAST discrete action tokens in planar_mode (pre-training style). "
             "Aligns subtask and action tokens in semantic space for better transfer."
    )
    parser.add_argument(
        "--fast_tokenizer_path", type=str, default=None,
        help="Path to FAST tokenizer (required if --use_fast_tokens). "
             "Can be 'physical-intelligence/fast' for official pretrained tokenizer."
    )
    parser.add_argument(
        "--action_expert_mode", action="store_true",
        help="Action expert training only (flow matching). Uses standard dataloader; skips text loss."
    )
    
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Pi05 train] Using device: {device}")

    # Sanity checks for mode flags
    if args.planar_mode and args.action_expert_mode:
        raise ValueError("Cannot enable both --planar_mode and --action_expert_mode simultaneously.")
    if args.planar_mode:
        # Force hierarchical data for subtask supervision
        args.hierarchical = True
        args.alpha = 0.0  # Only supervise text loss
        if args.use_fast_tokens:
            if args.fast_tokenizer_path is None:
                logger.warning(
                    "[Mode] --use_fast_tokens enabled but --fast_tokenizer_path not provided. "
                    "Using default 'physical-intelligence/fast'"
                )
                args.fast_tokenizer_path = "physical-intelligence/fast"
            logger.info(
                "[Mode] Planar mode with FAST tokens: "
                "high-level text CE + FAST action tokens (pre-training style). "
                "This aligns subtask and action tokens in semantic space for better transfer."
            )
        else:
            logger.info("[Mode] Planar mode enabled: high-level text CE only (alpha=0).")
    if args.action_expert_mode:
        # Ensure we do action-only training
        args.hierarchical = False
        logger.info("[Mode] Action expert mode enabled: action-only flow matching.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build config
    config = PI05Config(dtype=args.dtype)
    logger.info(f"Config: paligemma_variant={config.paligemma_variant}, "
                f"action_expert_variant={config.action_expert_variant}, "
                f"dtype={config.dtype}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        data_root=args.data_root,
        primary_camera=args.primary_camera,
        instruction_json=args.instruction_json,
        config=config,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        enable_hierarchical=args.hierarchical,
    )
    logger.info(f"Training dataset: {len(train_loader.dataset)} samples")
    if val_loader:
        logger.info(f"Validation dataset: {len(val_loader.dataset)} samples")
    
    if args.hierarchical:
        logger.info(f"Hierarchical training enabled (α = {args.alpha})")
        if args.planar_mode:
            logger.info("Planar mode: Only subtask CE supervised; action loss weight set to 0.")
    if args.action_expert_mode:
        logger.info("Action expert mode: Using action-only training loop.")

    # Build model and preprocessor
    model, preprocessor = build_model_and_preprocessor(
        config=config,
        device=device,
        dataset_stats_path=args.dataset_stats,
        use_augmentation=args.use_augmentation,
        fast_tokenizer_path=args.fast_tokenizer_path if args.use_fast_tokens else None,
    )

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optimizer - matching OpenPI's AdamW config
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),  # OpenPI default
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # Optional: Learning rate scheduler
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=1)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")

        if args.hierarchical:
            # Hierarchical training (Pi0.5 paper Eq. 1)
            train_losses = train_one_epoch_hierarchical(
                model=model,
                preprocessor=preprocessor,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                alpha=args.alpha,
                gradient_clip_norm=args.gradient_clip_norm,
                log_interval=args.log_interval,
            )
            train_loss = train_losses["total"]
        else:
            # Standard action-only training
            train_loss = train_one_epoch(
                model=model,
                preprocessor=preprocessor,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                gradient_clip_norm=args.gradient_clip_norm,
                log_interval=args.log_interval,
            )

        # Validation
        if val_loader:
            if args.hierarchical:
                val_losses = validate_hierarchical(model, preprocessor, val_loader, device, args.alpha)
                val_loss = val_losses["total"]
                logger.info(
                    f"[Epoch {epoch}] Validation: "
                    f"total = {val_loss:.6f}, "
                    f"action = {val_losses['action']:.6f}, "
                    f"text = {val_losses['text']:.6f}"
                )
            else:
                val_loss = validate(model, preprocessor, val_loader, device)
                logger.info(f"[Epoch {epoch}] Validation loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                logger.info(f"Saved best model to {best_path}")

        # Save checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "config": config,
                "hierarchical": args.hierarchical,
            }
            if args.hierarchical:
                save_dict["train_losses"] = train_losses
                save_dict["alpha"] = args.alpha
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
