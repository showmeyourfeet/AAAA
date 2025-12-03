import torch
import torch.optim as optim
import argparse
import os
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import logging
import json

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)  # Go up one level from HighLevelModel/ to src/
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from collections import OrderedDict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from hlp.dataset import (
    load_merged_data,
    load_splitted_data,
    load_composite_data,
    load_paired_stage_sequences,
    load_sequential_stage_sequences,
    AnnotationStageDataset,
    load_annotation_data,
)
from hlp.model import HighLevelModel


def calculate_hl_loss(logits, true_labels):
    """
    Calculate high-level loss with L1 distance weighting.
    Penalizes predictions that are further from the correct stage.
    
    Args:
        logits: Model output logits, shape (batch_size, num_classes)
        true_labels: Ground truth labels, shape (batch_size,)
    
    Returns:
        weighted_loss: Scalar tensor, distance-weighted cross-entropy loss
    """
    # Unreduced cross-entropy loss
    ce_loss_unreduced = torch.nn.functional.cross_entropy(logits, true_labels, reduction='none')
    
    # Calculate L1 distance between predicted and true labels
    with torch.no_grad():
        pred_indices = torch.argmax(logits, dim=-1)
        l1_distance = torch.abs(pred_indices - true_labels).float() + 1.0
    
    # Weight the loss by L1 distance
    weighted_loss = (ce_loss_unreduced * l1_distance).mean()
    
    return weighted_loss


def train(model, dataloader, optimizer, scheduler, device, logger=None, log_wandb=False, epoch: int | None = None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    batch_bar = tqdm(
        dataloader,
        desc=f"Train {epoch}" if epoch is not None else "Train",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in batch_bar:
        images, _, commands = batch
        images = images.to(device)

        optimizer.zero_grad()
        logits, temperature = model(images)

        # Convert ground truth command strings to indices using the pre-computed dictionary
        commands_idx = [
            model.command_to_index[
                cmd.replace("the back", "the bag").replace("mmove", "move")
            ]
            for cmd in commands
        ]
        commands_idx = torch.tensor(commands_idx, device=device)

        # Calculate distance-weighted loss
        loss = calculate_hl_loss(logits, commands_idx)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress bar live metrics
        try:
            temp_val = float(temperature.item()) if hasattr(temperature, "item") else float(temperature)
        except Exception:
            temp_val = 1.0
        batch_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            temp=f"{temp_val:.3f}",
        )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss


def evaluate(model, dataloader, device, epoch: int | None = None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        val_bar = tqdm(
            dataloader,
            desc=f"Val {epoch}" if epoch is not None else "Val",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in val_bar:
            images, _, commands = batch
            images = images.to(device)

            logits, temperature = model(images)

            # Convert ground truth command strings to indices using the pre-computed dictionary
            commands_idx = [
                model.command_to_index[cmd.replace("the back", "the bag")]
                for cmd in commands
            ]
            commands_idx = torch.tensor(commands_idx, device=device)

            # Calculate distance-weighted loss
            loss = calculate_hl_loss(logits, commands_idx)
            
            total_loss += loss.item()
            num_batches += 1
            val_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return avg_loss


def test(model, dataloader, device, current_epoch):
    model.eval()

    total_correct = 0
    total_predictions = 0

    # predicted_embeddings = []
    # gt_embeddings = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            images, command_embedding_gt, command_gt = batch
            images = images.to(device)

            logits, temperature = model(images)
            # Get nearest text for each prediction in the batch
            decoded_texts = model.decode_logits(logits, temperature)

            # predicted_embeddings.extend(predictions.cpu().numpy())
            # gt_embeddings.extend(command_embedding_gt.cpu().numpy())

            for i, (gt, pred) in enumerate(zip(command_gt, decoded_texts)):
                # Save incorrect prediction
                # if pred != gt:
                #     save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_incorrect_{idx}_{i}.jpg")
                #     save_combined_image(images[i].squeeze(0), gt, pred, save_path)
                #     if args.log_wandb:
                #         wandb.log({f"Incorrect Prediction": wandb.Image(save_path, caption=f"Epoch {current_epoch}, Batch {idx}, Image {i}")})
                # elif i < 5:
                #     save_path = os.path.join(ckpt_dir, "predictions", f"epoch_{current_epoch}_correct_{idx}_{i}.jpg")
                #     save_combined_image(images[i].squeeze(0), gt, pred, save_path)

                total_correct += int(pred == gt)
                total_predictions += 1
                print(f"Ground truth: {gt} \t Predicted text: {pred}")

    # Visualize embeddings
    # tsne_visualize(predicted_embeddings, gt_embeddings, candidate_embeddings, current_epoch)

    success_rate = total_correct / total_predictions
    print(f"Epoch {current_epoch}: Success Rate = {success_rate * 100:.2f}%")

    # wandb removed

    return success_rate


def latest_checkpoint(ckpt_dir):
    """
    Returns the latest checkpoint file from the given directory.
    Priority:
    1. latest_checkpoint.ckpt (rolling backup, most recent)
    2. epoch_*.ckpt (periodic backups every 100 epochs)
    """
    # First, check for latest_checkpoint.ckpt (rolling backup)
    latest_ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint.ckpt")
    if os.path.exists(latest_ckpt_path):
        try:
            checkpoint = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                epoch_idx = checkpoint['epoch']
                return latest_ckpt_path, epoch_idx
        except Exception as e:
            print(f"Warning: Failed to load latest_checkpoint.ckpt: {e}")
    
    # Fallback: Find the latest epoch_*.ckpt
    all_ckpts = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("epoch_") and f.endswith(".ckpt")
    ]
    epoch_numbers = [int(f.split("_")[1].split(".")[0]) for f in all_ckpts]

    # If no valid checkpoints are found, return None
    if not epoch_numbers:
        return None, None

    latest_idx = max(epoch_numbers)
    return os.path.join(ckpt_dir, f"epoch_{latest_idx}.ckpt"), latest_idx


def load_candidate_texts(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Extract the instruction (text before the colon), strip whitespace, and then strip quotation marks
        candidate_texts = [line.split(":")[0].strip().strip("'\"") for line in lines]
    return candidate_texts


def save_combined_image(image, gt_text, pred_text, save_path=None):
    # image = image[:, :, [2, 1, 0]]

    # Extract first frame t=0 and concatenate across width
    combined_image = torch.cat([image[0, i] for i in range(image.shape[1])], dim=-1)

    # Convert to PIL image
    combined_image_pil = transforms.ToPILImage()(combined_image)

    # Create a blank canvas to add text
    canvas = Image.new(
        "RGB", (combined_image_pil.width, combined_image_pil.height + 100), "black"
    )
    canvas.paste(combined_image_pil, (0, 100))

    # Add GT and predicted text
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", 30
    )
    draw.text((10, 10), "GT: " + gt_text, font=font, fill="white")
    draw.text((10, 50), "Pred: " + pred_text, font=font, fill="red")

    if save_path is not None:
        canvas.save(save_path)
    else:
        return canvas


def tsne_visualize(predicted_embeddings, gt_embeddings, candidate_embeddings, epoch):
    # Convert lists to numpy arrays
    predicted_embeddings = np.array(predicted_embeddings)
    gt_embeddings = np.array(gt_embeddings)

    assert (
        predicted_embeddings.shape == gt_embeddings.shape
    ), "The number of predicted and ground truth embeddings do not match."

    # Stack embeddings and apply t-SNE
    all_embeddings = np.vstack(
        [predicted_embeddings, gt_embeddings, candidate_embeddings.cpu().numpy()]
    )
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split the 2D embeddings back
    predicted_2d = embeddings_2d[: len(predicted_embeddings)]
    gt_2d = embeddings_2d[
        len(predicted_embeddings) : len(predicted_embeddings) + len(gt_embeddings)
    ]
    candidate_2d = embeddings_2d[len(predicted_embeddings) + len(gt_embeddings) :]

    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.scatter(
        candidate_2d[:, 0], candidate_2d[:, 1], marker="o", color="g", label="Dataset"
    )
    plt.scatter(gt_2d[:, 0], gt_2d[:, 1], marker="o", color="b", label="Ground Truth")
    plt.scatter(
        predicted_2d[:, 0], predicted_2d[:, 1], marker="o", color="r", label="Predicted"
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.title(f"t-SNE Visualization of Embeddings (Epoch {epoch})")
    plt.legend()

    # Save with the epoch in the filename
    image_save_path = os.path.join(ckpt_dir, f"embeddings_tsne_epoch_{epoch}.png")
    plt.savefig(image_save_path)

    # wandb removed


def load_candidate_texts_and_embeddings(dataset_dirs, device=torch.device("cuda"), stage_embeddings_file=None, stage_texts_file=None):
    """
    Load candidate texts and embeddings.
    Supports both traditional format (from dataset_dirs) and splitted format (from stage_embeddings_file).
    """
    if stage_embeddings_file and os.path.exists(stage_embeddings_file):
        # Load from splitted format
        candidate_texts = []
        candidate_embeddings = []
        
        with open(stage_embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get('stage_embeddings', data)
        
        if isinstance(entries, list):
            for e in entries:
                if 'stage' in e and 'embedding' in e:
                    stage_idx = int(e['stage'])
                    embedding = torch.tensor(e['embedding']).float().to(device).squeeze()
                    candidate_embeddings.append(embedding)
                    # Prefer text inside the same entry if available
                    if 'text' in e and isinstance(e['text'], str) and len(e['text']) > 0:
                        text = e['text']
                    else:
                        # Fallback: read from stage_texts_file, else default
                        if stage_texts_file and os.path.exists(stage_texts_file):
                            with open(stage_texts_file, 'r', encoding='utf-8') as f2:
                                texts_data = json.load(f2)
                            # try structured 'stage_texts'; otherwise allow dict with numeric keys
                            texts_entries = texts_data.get('stage_texts', {})
                            if isinstance(texts_entries, dict):
                                text = texts_entries.get(str(stage_idx), f"stage_{stage_idx}")
                            else:
                                # last resort default
                                text = f"stage_{stage_idx}"
                        else:
                            text = f"stage_{stage_idx}"
                    candidate_texts.append(text)
        elif isinstance(entries, dict):
            # Filter and sort entries by stage index to ensure consistent order
            # This is critical for the L1 distance loss function which assumes index corresponds to stage order
            valid_entries = []
            for k, v in entries.items():
                try:
                    idx = int(k)
                    valid_entries.append((idx, str(k), v))
                except (ValueError, TypeError):
                    continue
            
            # Sort by stage index
            valid_entries.sort(key=lambda x: x[0])

            for stage_idx, stage_idx_str, embedding in valid_entries:
                embedding_tensor = torch.tensor(embedding).float().to(device).squeeze()
                candidate_embeddings.append(embedding_tensor)
                
                if stage_texts_file and os.path.exists(stage_texts_file):
                    with open(stage_texts_file, 'r', encoding='utf-8') as f2:
                        texts_data = json.load(f2)
                    texts_entries = texts_data.get('stage_texts', texts_data)
                    if isinstance(texts_entries, dict):
                        # Try to get text by stage_idx (as int or str)
                        text = texts_entries.get(stage_idx_str, texts_entries.get(str(stage_idx), f"stage_{stage_idx}"))
                    else:
                        text = f"stage_{stage_idx}"
                else:
                    text = f"stage_{stage_idx}"
                candidate_texts.append(text)
        
        if candidate_embeddings:
            candidate_embeddings = torch.stack(candidate_embeddings).to(device)
        else:
            raise ValueError(f"No valid embeddings found in {stage_embeddings_file}")
        
        return candidate_texts, candidate_embeddings
    
    # Traditional format: load from dataset_dirs
    candidate_texts = []
    candidate_embeddings = []

    for dataset_dir in dataset_dirs:
        embeddings_path = os.path.join(
            dataset_dir, "candidate_embeddings_distilbert.npy"
        )
        # Load pre-computed embeddings
        candidate_embedding = (
            torch.tensor(np.load(embeddings_path).astype(np.float32))
            .to(device)
            .squeeze()
        )
        candidate_embeddings.append(candidate_embedding)
        candidate_texts_path = os.path.join(dataset_dir, "count.txt")
        current_candidate_texts = load_candidate_texts(candidate_texts_path)
        candidate_texts.extend(current_candidate_texts)
    candidate_embeddings = torch.cat(candidate_embeddings, dim=0).to(device)

    def remove_duplicates(candidate_texts, candidate_embeddings):
        unique_entries = OrderedDict()

        for text, embedding in zip(candidate_texts, candidate_embeddings):
            if text not in unique_entries:
                unique_entries[text] = embedding

        # Rebuild the lists without duplicates
        filtered_texts = list(unique_entries.keys())
        filtered_embeddings = torch.stack(list(unique_entries.values()))

        return filtered_texts, filtered_embeddings

    candidate_texts, candidate_embeddings = remove_duplicates(
        candidate_texts, candidate_embeddings
    )
    return candidate_texts, candidate_embeddings


def build_HighLevelModel(dataset_dirs, history_len, device, stage_embeddings_file=None, stage_texts_file=None, train_image_encoder=False):
    # Load candidate texts and embeddings
    candidate_texts, candidate_embeddings = load_candidate_texts_and_embeddings(
        dataset_dirs, device=device, stage_embeddings_file=stage_embeddings_file, stage_texts_file=stage_texts_file
    )
    command_to_index = {command: index for index, command in enumerate(candidate_texts)}

    # Build model
    model = HighLevelModel(
        device=device,
        history_len=history_len,
        candidate_embeddings=candidate_embeddings,
        candidate_texts=candidate_texts,
        command_to_index=command_to_index,
        train_image_encoder=train_image_encoder,
    ).to(device)
    return model


if __name__ == "__main__":
    # memory monitor removed

    parser = argparse.ArgumentParser(description="Train and evaluate command prediction model using CLIP.")
    # task_name removed; prefer explicit dataset roots
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    # wandb removed
    parser.add_argument('--gpu', action='store', type=int, help='gpu', default=0)
    parser.add_argument('--history_len', action='store', type=int, help='history_len', default=1)
    parser.add_argument('--prediction_offset', action='store', type=int, help='prediction_offset', default=20)
    parser.add_argument('--history_skip_frame', action='store', type=int, help='history_skip_frame', default=50)
    parser.add_argument('--test_only', action='store_true', help='Test the model using the latest checkpoint and exit')
    parser.add_argument('--random_crop', action='store_true')
    parser.add_argument('--use_augmentation', action='store_true', help='Enable data augmentation during training')
    parser.add_argument('--sync_sequence_augmentation', action='store_true',
                        help='Sync augmentation across the history frames of a sample (SequenceDataset only)')
    parser.add_argument('--image_size', action='store', type=int, help='Image size for augmentation', default=224)
    parser.add_argument('--dagger_ratio', action='store', type=float, help='dagger_ratio', default=None)
    parser.add_argument('--use_splitted', action='store_true', help='Use splitted dataset format (stage*/run_*)')
    parser.add_argument('--use_composite', action='store_true', help='Use composite dataset format (randomly select one run per stage and concatenate)')
    parser.add_argument('--splitted_root', action='store', type=str, help='Root directory for splitted dataset (training data)', default=None)
    parser.add_argument('--stage_embeddings_file', action='store', type=str, help='JSON file with stage embeddings', default=None)
    parser.add_argument('--stage_texts_file', action='store', type=str, help='JSON file with stage texts', default=None)
    parser.add_argument('--val_splitted_root', action='store', type=str, help='Root directory for validation (splitted format). If unset, will split from training data.', default=None)
    parser.add_argument('--sampling_strategy', action='store', type=str, choices=['balanced', 'random', 'sequential'], default='balanced', 
                        help='Sampling strategy for composite dataset: balanced (default, ensures equal run usage), random (fully random), sequential (for validation)')
    parser.add_argument('--max_combinations', action='store', type=int, default=None,
                        help='Maximum number of run combinations to generate (only for composite dataset). If not set, use --n_repeats instead.')
    parser.add_argument('--n_repeats', action='store', type=int, default=None,
                        help='Number of times each run should be used. Automatically calculates max_combinations = n_repeats × min_runs_per_stage. '
                             'Use this for intuitive control (e.g., --n_repeats 10 means each run used 10 times). '
                             'Ignored if --max_combinations is set.')
    # Paired adjacent-stage training (concatenate two recordings)
    parser.add_argument('--use_paired_sequences', action='store', type=bool, default=False,
                        help='Use paired adjacent-stage sequences for training (splitted format only)')
    parser.add_argument('--max_run', action='store', type=int, default=None,
                        help='Max runs per (i,i+1) pair per epoch (None = all)')
    # traditional format replacements
    parser.add_argument('--dataset_dirs', nargs='+', type=str, help='List of dataset directories (traditional format)')
    parser.add_argument('--camera_names', nargs='+', type=str, default=['left_frame','right_frame'], help='Camera names')
    # Annotation-based continuous runs (run_XXX with annotations.json)
    parser.add_argument('--use_annotation_dataset', action='store_true',
                        help='Use annotation-based dataset format (run_xx/frames + annotations.json)')
    parser.add_argument('--annotation_root', type=str, default=None,
                        help='Root directory containing run_*/ with frames/ and annotations.json')
    parser.add_argument('--traverse_full_trajectory', action='store_true',
                        help='For annotation dataset: traverse full trajectories deterministically instead of random sampling')
    # Image encoder training control
    parser.add_argument('--train_image_encoder', action='store_true', help='Enable training of the image encoder (default: frozen)')
    parser.add_argument('--save_ckpt_every', type=int, default=100,
                        help='Interval size for saving the best checkpoint within each interval [i*n, (i+1)*n) based on val loss (default: 100)')

    args = parser.parse_args()

    # Setting seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device setting
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    ckpt_dir = args.ckpt_dir
    dagger_ratio = args.dagger_ratio

    # Data loading
    if args.use_annotation_dataset:
        # New annotation-format dataset: example_datasets/run_XXX with annotations.json
        assert args.annotation_root is not None, "--annotation_root must be provided when --use_annotation_dataset is set"
        assert not args.use_splitted and not args.use_composite, "--use_annotation_dataset is mutually exclusive with --use_splitted/--use_composite"

        train_dataloader, val_dataloader, test_dataloader = load_annotation_data(
            root_dir=args.annotation_root,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_skip_frame=args.history_skip_frame,
            traverse_full_trajectory=args.traverse_full_trajectory,
            stage_embeddings_file=args.stage_embeddings_file,
            stage_texts_file=args.stage_texts_file,
            use_augmentation=args.use_augmentation,
            image_size=args.image_size,
            drop_initial_frames=1,
            train_ratio=0.85,
            val_ratio=0.15,
            seed=args.seed,
        )

        # Build model with stage-level candidate texts/embeddings
        model = build_HighLevelModel(
            dataset_dirs=[],  # not used in annotation mode
            history_len=args.history_len,
            device=device,
            stage_embeddings_file=args.stage_embeddings_file,
            stage_texts_file=args.stage_texts_file,
            train_image_encoder=args.train_image_encoder,
        )

    elif args.use_splitted or args.use_composite:
        # Use splitted or composite dataset format
        assert args.splitted_root is not None, "--splitted_root must be provided when --use_splitted or --use_composite is set"
        assert args.stage_embeddings_file is not None, "--stage_embeddings_file must be provided when --use_splitted or --use_composite is set"
        
        # Get camera names from task config or use default
        camera_names = args.camera_names
        
        if args.use_composite:
            # Use composite dataset format (select one run per stage and concatenate)
            train_dataloader, val_dataloader, test_dataloader = load_composite_data(
                root_dir=args.splitted_root,
                camera_names=camera_names,
                batch_size_train=args.batch_size,
                batch_size_val=args.batch_size,
                history_len=args.history_len,
                prediction_offset=args.prediction_offset,
                history_skip_frame=args.history_skip_frame,
                random_crop=args.random_crop,
                stage_embeddings_file=args.stage_embeddings_file,
                stage_texts_file=args.stage_texts_file,
                dagger_ratio=dagger_ratio,
                train_subdir=None,  # root_dir is training directory directly
                val_subdir=None,    # use val_root instead
                val_root=args.val_splitted_root,  # Separate validation directory
                use_augmentation=args.use_augmentation,
                image_size=args.image_size,
                sampling_strategy=args.sampling_strategy,
                max_combinations=args.max_combinations,
                n_repeats=args.n_repeats,
                sync_sequence_augmentation=args.sync_sequence_augmentation,
            )
        elif args.use_paired_sequences:
            # Use paired adjacent-stage sequences for training; keep validation/test as None for now
            train_dataloader = load_paired_stage_sequences(
                root_dir=args.splitted_root,
                camera_names=camera_names,
                batch_size=args.batch_size,
                history_len=args.history_len,
                skip_frame=args.history_skip_frame,
                offset=args.prediction_offset,
                stage_embeddings_file=args.stage_embeddings_file,
                stage_texts_file=args.stage_texts_file,
                max_run=args.max_run,
                use_augmentation=args.use_augmentation,
                image_size=args.image_size,
                rng_seed=args.seed,  # initial pairing
            )
            # For validation/test, use deterministic sequential concatenation per run_name
            val_root = args.val_splitted_root or args.splitted_root
            val_dataloader = load_sequential_stage_sequences(
                root_dir=val_root,
                camera_names=camera_names,
                batch_size=args.batch_size,
                history_len=args.history_len,
                skip_frame=args.history_skip_frame,
                offset=args.prediction_offset,
                stage_embeddings_file=args.stage_embeddings_file,
                stage_texts_file=args.stage_texts_file,
                use_augmentation=False,
                image_size=args.image_size,
            )
            test_dataloader = val_dataloader
        else:
            train_dataloader, val_dataloader, test_dataloader = load_splitted_data(
                root_dir=args.splitted_root,
                camera_names=camera_names,
                batch_size_train=args.batch_size,
                batch_size_val=args.batch_size,
                history_len=args.history_len,
                prediction_offset=args.prediction_offset,
                history_skip_frame=args.history_skip_frame,
                random_crop=args.random_crop,
                stage_embeddings_file=args.stage_embeddings_file,
                stage_texts_file=args.stage_texts_file,
                dagger_ratio=dagger_ratio,
                use_augmentation=args.use_augmentation,
                image_size=args.image_size,
            )
        
        # Build model with splitted/composite format
        model = build_HighLevelModel(
            dataset_dirs=[],  # Empty for splitted/composite format
            history_len=args.history_len,
            device=device,
            stage_embeddings_file=args.stage_embeddings_file,
            stage_texts_file=args.stage_texts_file,
            train_image_encoder=args.train_image_encoder,
        )
    else:
        # Traditional format: use explicit dataset_dirs and camera_names
        assert args.dataset_dirs is not None and len(args.dataset_dirs) > 0, "--dataset_dirs must be provided"
        dataset_dirs = args.dataset_dirs
        # Fallbacks (dataset.py expects traditional loaders to know sizes; we set to 1 to load directly)
        num_episodes_list = [1 for _ in dataset_dirs]
        camera_names = args.camera_names
        train_dataloader, val_dataloader, test_dataloader = load_merged_data(
            dataset_dirs=dataset_dirs,
            num_episodes_list=num_episodes_list,
            camera_names=camera_names,
            batch_size_train=args.batch_size,
            batch_size_val=args.batch_size,
            history_len=args.history_len,
            prediction_offset=args.prediction_offset,
            history_skip_frame=args.history_skip_frame,
            random_crop=args.random_crop,
            dagger_ratio=dagger_ratio,
            use_augmentation=args.use_augmentation,
            image_size=args.image_size,
            sync_sequence_augmentation=args.sync_sequence_augmentation,
        )

        model = build_HighLevelModel(
            dataset_dirs, 
            args.history_len, 
            device,
            train_image_encoder=args.train_image_encoder,
        )
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    
    # Learning rate scheduler with warmup and cosine annealing
    total_steps = args.num_epochs
    warmup_steps = max(1, total_steps // 100)  # 1% warmup
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        ],
        milestones=[warmup_steps]
    )

    # WandB removed

    # Log dataset sizes
    def _dataset_size(dataloader):
        try:
            return len(dataloader.dataset)
        except Exception:
            try:
                return len(dataloader)
            except Exception:
                return None
    train_size = _dataset_size(train_dataloader)
    val_size = _dataset_size(val_dataloader) if 'val_dataloader' in locals() and val_dataloader is not None else None
    test_size = _dataset_size(test_dataloader) if 'test_dataloader' in locals() and test_dataloader is not None else None
    logging.info(f"  Train Samples: {train_size if train_size is not None else 'N/A'}")
    logging.info(f"  Val Samples: {val_size if val_size is not None else 'N/A'}")
    logging.info(f"  Test Samples: {test_size if test_size is not None else 'N/A'}")

    # Set up logging (console + file)
    logger = logging.getLogger("HighLevelModel_train")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler for local log
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(ckpt_dir, "train.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # Avoid duplicate logs to root logger
    logger.propagate = False
    
    # Log training configuration
    logger.info("=" * 80)
    logger.info("Training Configuration:")
    logger.info(f"  Checkpoint Directory: {ckpt_dir}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Number of Epochs: {args.num_epochs}")
    logger.info(f"  History Length: {args.history_len}")
    logger.info(f"  Prediction Offset: {args.prediction_offset}")
    logger.info(f"  History Skip Frame: {args.history_skip_frame}")
    logger.info(f"  Train Image Encoder: {args.train_image_encoder}")
    logger.info(f"  Use Splitted Dataset: {args.use_splitted}")
    logger.info(f"  Use Composite Dataset: {args.use_composite}")
    if args.use_composite:
        logger.info(f"  Sampling Strategy: {args.sampling_strategy}")
        if args.max_combinations is not None:
            logger.info(f"  Max Combinations: {args.max_combinations}")
        elif args.n_repeats is not None:
            logger.info(f"  N Repeats: {args.n_repeats} (max_combinations will be auto-calculated)")
        else:
            logger.info(f"  Max Combinations: 50000 (default)")
    logger.info(f"  Use Data Augmentation: {args.use_augmentation}")
    if args.use_augmentation:
        logger.info(f"  Image Size: {args.image_size}")
    if args.use_splitted or args.use_composite:
        logger.info(f"  Training Root: {args.splitted_root}")
        if args.val_splitted_root:
            logger.info(f"  Validation Root: {args.val_splitted_root}")
        logger.info(f"  Stage Embeddings File: {args.stage_embeddings_file}")
    logger.info("=" * 80)
    # Also report dataset sizes
    def _dataset_size(dataloader):
        try:
            return len(dataloader.dataset)
        except Exception:
            try:
                return len(dataloader)
            except Exception:
                return None
    train_size = _dataset_size(train_dataloader)
    val_size = _dataset_size(val_dataloader) if 'val_dataloader' in locals() and val_dataloader is not None else None
    test_size = _dataset_size(test_dataloader) if 'test_dataloader' in locals() and test_dataloader is not None else None
    logger.info(f"  Train Samples: {train_size if train_size is not None else 'N/A'}")
    logger.info(f"  Val Samples: {val_size if val_size is not None else 'N/A'}")
    logger.info(f"  Test Samples: {test_size if test_size is not None else 'N/A'}")

    # Initialize training state variables
    best_val_loss = float('inf')
    best_ckpt_path = None
    best_epoch = -1

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
        latest_idx = 0
    else:
        # Load the most recent checkpoint if available
        latest_ckpt, latest_idx = latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            logger.info(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
            
            # Handle both old format (state_dict only) and new format (dict with keys)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format with full training state
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if 'best_val_loss' in checkpoint:
                    best_val_loss = checkpoint['best_val_loss']
                    logger.info(f"Restored best_val_loss: {best_val_loss:.5f}")
                if 'best_epoch' in checkpoint:
                    best_epoch = checkpoint.get('best_epoch', -1)
                logger.info("Loaded full training state (model + optimizer + scheduler)")
            else:
                # Old format: only state_dict
                model.load_state_dict(checkpoint)
                logger.info("Loaded model weights only (old format)")
        else:
            logger.info("No checkpoint found. Starting from scratch.")
            latest_idx = 0

    predictions_dir = os.path.join(ckpt_dir, "predictions")
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    if args.test_only:
        if 'test_dataloader' in locals() and test_dataloader is not None:
            test(model, test_dataloader, device, latest_idx)
        else:
            logger.info("test_only is set but no test_dataloader is available; exiting.")
        exit()

    # Start training from loaded or initial state
    save_ckpt_every = args.save_ckpt_every
    interval_best_val = float('inf')
    interval_best_epoch = -1
    interval_best_ckpt_path = None
    current_interval_idx = None

    logger.info(f"Starting training from epoch {latest_idx} to {args.num_epochs}")
    
    pbar_epochs = tqdm(range(latest_idx, args.num_epochs), desc="Epochs", leave=True)
    for epoch in pbar_epochs:
        # Rebuild train dataloader each epoch to randomize pairings (if enabled)
        if args.use_splitted and args.use_paired_sequences:
            train_dataloader = load_paired_stage_sequences(
                root_dir=args.splitted_root,
                camera_names=camera_names,
                batch_size=args.batch_size,
                history_len=args.history_len,
                skip_frame=args.history_skip_frame,
                offset=args.prediction_offset,
                stage_embeddings_file=args.stage_embeddings_file,
                stage_texts_file=args.stage_texts_file,
                max_run=args.max_run,
                use_augmentation=args.use_augmentation,
                image_size=args.image_size,
                rng_seed=int(args.seed + epoch),
            )
        # Test the model and log success rate every 200 epochs (if test_dataloader is available)
        if (
            epoch % 200 == 0
            and (epoch > 0 or dagger_ratio is not None)
            and 'test_dataloader' in locals()
            and test_dataloader is not None
        ):
            test_success_rate = test(model, test_dataloader, device, epoch)
            if logger:
                logger.info(f"Epoch {epoch}: Test Success Rate = {test_success_rate * 100:.2f}%")

        # Training
        train_loss = train(model, train_dataloader, optimizer, scheduler, device, logger=logger, log_wandb=False, epoch=epoch)
        
        # Step scheduler after training
        scheduler.step()
        
        # Validation
        val_loss = None
        if val_dataloader is not None and dagger_ratio is None:
            val_loss = evaluate(model, val_dataloader, device, epoch=epoch)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar
        postfix_dict = {"Train Loss": f"{train_loss:.5f}"}
        if val_loss is not None:
            postfix_dict["Val Loss"] = f"{val_loss:.5f}"
        pbar_epochs.set_postfix(postfix_dict)

        # Logging in unified format (compatible with llp/train.py)
        if logger:
            logger.info(f"Epoch {epoch}: Train loss: {train_loss:.5f}")
            logger.info(f"epoch: {epoch} loss: {train_loss:.5f} lr: {current_lr:.5e}")
            if val_loss is not None:
                logger.info(f"Epoch {epoch}: Val loss: {val_loss:.5f}")
        else:
            tqdm.write(f"Train loss: {train_loss:.5f}" + 
                      (f", Val loss: {val_loss:.5f}" if val_loss is not None else ""))

        # wandb removed

        # Save best model based on validation loss
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_ckpt_path = os.path.join(ckpt_dir, f"best_model.ckpt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, best_ckpt_path)
            logger.info(f"✓ New best model! Epoch {epoch}: Val loss = {best_val_loss:.5f} -> {best_ckpt_path}")

        # Interval-based best checkpoint saving (based on val_loss)
        if val_loss is not None:
            this_interval_idx = epoch // save_ckpt_every
            if current_interval_idx is None or this_interval_idx != current_interval_idx:
                interval_best_val = float('inf')
                interval_best_epoch = -1
                if interval_best_ckpt_path and os.path.exists(interval_best_ckpt_path):
                    # keep the best checkpoint from previous interval; no deletion
                    pass
                interval_best_ckpt_path = None
                current_interval_idx = this_interval_idx

            if val_loss < interval_best_val:
                interval_best_val = val_loss
                if interval_best_ckpt_path and os.path.exists(interval_best_ckpt_path):
                    os.remove(interval_best_ckpt_path)
                    if logger:
                        logger.info(f"Deleted old interval checkpoint: {interval_best_ckpt_path}")
                interval_best_epoch = epoch
                interval_start = this_interval_idx * save_ckpt_every
                interval_end = (this_interval_idx + 1) * save_ckpt_every
                interval_best_ckpt_path = os.path.join(
                    ckpt_dir, f"interval_{this_interval_idx}_epoch_{epoch}_best.ckpt"
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                }, interval_best_ckpt_path)
                if logger:
                    logger.info(
                        f"Saved interval [{interval_start}-{interval_end}) best checkpoint "
                        f"(epoch {epoch}, val_loss: {val_loss:.5f}) to {interval_best_ckpt_path}"
                    )
                else:
                    tqdm.write(
                        f"Saved interval [{interval_start}-{interval_end}) best checkpoint "
                        f"(epoch {epoch}, val_loss: {val_loss:.5f})"
                    )

        # Save latest checkpoint every epoch (rolling backup)
        # This allows resuming from any epoch, not just multiples of 100
        latest_ckpt_path = os.path.join(ckpt_dir, "latest_checkpoint.ckpt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        }, latest_ckpt_path)

        # Save a checkpoint every 100 epochs
        if epoch % save_ckpt_every == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch}.ckpt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
            }, ckpt_path)
            if logger:
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # Pruning: this removes the checkpoint save_ckpt_every epochs behind the current one
            # except for the ones at multiples of prune_freq epochs
            # prune_freq = 300
            # prune_epoch = epoch - save_ckpt_every
            # if prune_epoch % prune_freq != 0:
            #     prune_path = os.path.join(ckpt_dir, f"epoch_{prune_epoch}.ckpt")
            #     if os.path.exists(prune_path):
            #         os.remove(prune_path)
            #         if logger:
            #             logger.info(f"Pruned old checkpoint: {prune_path}")

    # Save final model
    final_ckpt_path = os.path.join(ckpt_dir, "final_model.ckpt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    }, final_ckpt_path)
    logger.info(f"Saved final model to {final_ckpt_path}")
    
    # Summary
    logger.info("=" * 80)
    logger.info("Training Summary:")
    if best_ckpt_path:
        logger.info(f"  Best Model: Epoch {best_epoch}, Val loss = {best_val_loss:.5f}")
        logger.info(f"  Best Model Path: {best_ckpt_path}")
    logger.info(f"  Final Model Path: {final_ckpt_path}")
    logger.info("=" * 80)

    # wandb removed
