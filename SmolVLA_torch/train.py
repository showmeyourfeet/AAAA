import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset import SmolVLADataset, smolvla_collate_fn
from policy import SmolVLA


def train_smolvla(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # Dataset & DataLoader
    dataset = SmolVLADataset(
        root_dir=args.root_dir,
        camera_names=args.camera_names,
        chunk_size=args.chunk_size,
        instruction_json=args.instruction_json,
        image_size=args.image_size,
        use_augmentation=args.use_augmentation,
        training=True,
        use_state=not args.no_state,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=smolvla_collate_fn,
    )

    # Peek one batch to infer dims
    sample_images, sample_state, sample_actions, _, _ = next(iter(train_loader))
    # 只用第一路相机，形状 [B, num_cams, C, H, W] -> [B, C, H, W]
    num_cams = sample_images.shape[1]
    action_dim = sample_actions.shape[-1]
    state_dim = sample_state.shape[-1]

    # Tokenizer for language
    tokenizer = AutoTokenizer.from_pretrained(args.vlm_model_id)

    # Model
    model = SmolVLA(
        action_dim=action_dim,
        state_dim=state_dim,
        chunk_size=args.chunk_size,
        vlm_model_id=args.vlm_model_id,
        device=device,
        prefix_length=args.prefix_length,
        max_state_dim=state_dim,
        max_action_dim=action_dim,
        resize_imgs_with_padding=(args.image_height, args.image_width) if args.image_height and args.image_width else None,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in loader:
            images, states, actions, is_pad, texts = batch

            # 只用第一路相机
            images = images[:, 0]  # [B, C, H, W]

            images = images.to(device)
            states = states.to(device)
            actions = actions.to(device)

            # 文本 -> tokens
            enc = tokenizer(
                list(texts),
                padding="longest",
                truncation=True,
                max_length=args.tokenizer_max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)

            optimizer.zero_grad()
            loss = model(images, input_ids, states, actions)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            loader.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"[Epoch {epoch+1}] avg_loss={avg_loss:.6f}")

        # 简单保存 checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f"smolvla_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Splitted dataset root (stage*/run*/data.txt)")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--instruction_json", type=str, default="utils/instruction_new.json")
    parser.add_argument("--vlm_model_id", type=str, default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--camera_names", nargs="*", default=["left_frame", "right_frame"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_height", type=int, default=None)
    parser.add_argument("--image_width", type=int, default=None)
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--no_state", action="store_true")
    parser.add_argument("--prefix_length", type=int, default=-1)
    parser.add_argument("--tokenizer_max_length", type=int, default=48)

    args = parser.parse_args()
    train_smolvla(args)


if __name__ == "__main__":
    main()


