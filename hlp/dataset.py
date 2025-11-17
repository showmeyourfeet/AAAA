import numpy as np
import torch
import os
import h5py
import cv2
import json
import re
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
import albumentations as A

from llp.utils import crop_resize, random_crop
from llp.dataset import DAggerSampler


class ImagePreprocessor:
    """Image preprocessor with data augmentation support.
    
    Based on the ImagePreprocessor from srt-h/utils.py, adapted for HighLevelModel training.
    """
    def __init__(self, image_size: int = 224, use_augmentation: bool = True, normalize: bool = False):
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.normalize = normalize

        # Define Albumentations enhancement process
        if use_augmentation:
            self.albumentations_transform = A.Compose([
                # Don't resize here, keep original size for now
                # Use multiple enhancement methods
                A.OneOf([
                    A.Rotate(limit=[-10, 10], p=0.5),
                    A.Affine(rotate=[-10, 10], scale=[0.9, 1.1], translate_percent=[-0.1, 0.1], shear=[-10, 10], p=0.8),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                ], p=0.5),
                A.CoarseDropout(num_holes_range=[1, 3], hole_height_range=[0.1, 0.2], hole_width_range=[0.1, 0.2], p=0.5)
            ])
        else:
            # Evaluation: no augmentation
            self.albumentations_transform = None

        # # Custom center crop
        # self.custom_crop = CustomCenterCrop()

        # Shared transforms
        self.to_tensor = T.ToTensor()
        if normalize:
            self.normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            self.normalize_transform = None

    def augment_image(self, image_np: np.ndarray, training: bool) -> np.ndarray:
        """Apply augmentation to numpy array image."""
        if training and self.use_augmentation and self.albumentations_transform is not None:
            # Apply Albumentations enhancement
            augmented = self.albumentations_transform(image=image_np)
            return augmented['image']
        return image_np

    def process(self, image: Image.Image, training: bool = False) -> torch.Tensor:
        """Process a single PIL image with optional augmentation and normalization."""
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Apply augmentation
        image_np = self.augment_image(image_np, training)
        
        # Convert to tensor
        image_tensor = self.to_tensor(image_np)
        
        # Normalize if requested
        if self.normalize_transform is not None:
            image_tensor = self.normalize_transform(image_tensor)
        
        return image_tensor


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        history_len=5,
        prediction_offset=10,
        history_skip_frame=1,
        random_crop=False,
        use_augmentation=False,
        training=True,
        image_size=224,
    ):
        super().__init__()
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.random_crop = random_crop
        self.use_augmentation = use_augmentation
        self.training = training
        self.image_size = image_size
        
        # Initialize ImagePreprocessor if augmentation is enabled
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation
            )

    def __len__(self):
        return len(self.episode_ids)

    def get_command_for_ts(self, episode_data, target_ts):
        for segment in episode_data:
            if segment["start_timestep"] <= target_ts <= segment["end_timestep"]:
                return torch.tensor(segment["embedding"]).squeeze(), segment["command"]
        return None, None

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        encoded_json_path = os.path.join(
            self.dataset_dir, f"episode_{episode_id}_encoded_distilbert.json"
        )

        with open(encoded_json_path, "r") as f:
            episode_data = json.load(f)

        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)

            # Sample a random curr_ts and compute the start_ts and target_ts
            total_timesteps = root["/action"].shape[0]
            prediction_offset = self.prediction_offset
            try:
                curr_ts = np.random.randint(
                    self.history_len * self.history_skip_frame,
                    total_timesteps - prediction_offset,
                )
                start_ts = curr_ts - self.history_len * self.history_skip_frame
                target_ts = curr_ts + prediction_offset
            except ValueError:
                # sample a different episode in range len(self.episode_ids)
                return self.__getitem__(np.random.randint(0, len(self.episode_ids)))

            # Retrieve the language embedding for the target_ts
            command_embedding, command_gt = self.get_command_for_ts(
                episode_data, target_ts
            )
            if command_embedding is None:
                try:
                    return self.__getitem__((index + 1) % len(self.episode_ids))
                except RecursionError:
                    print(
                        f"RecursionError: Could not find embedding for episode_id {episode_id} and target_ts {target_ts}."
                    )
                    import ipdb; ipdb.set_trace()

            # Construct the image sequences for the desired timesteps
            image_sequence = []
            for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
                image_dict = {}
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f"/observations/images/{cam_name}"][ts]
                    if compressed:
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)
                        if cam_name == "cam_high":
                            image_dict[cam_name] = crop_resize(image_dict[cam_name])
                        if self.random_crop:
                            image_dict[cam_name] = random_crop(image_dict[cam_name])
                    image_dict[cam_name] = cv2.cvtColor(
                        image_dict[cam_name], cv2.COLOR_BGR2RGB
                    )
                    
                    # Apply data augmentation if enabled
                    if self.use_augmentation:
                        image_dict[cam_name] = self.image_preprocessor.augment_image(
                            image_dict[cam_name], self.training
                        )
                
                all_cam_images = [
                    image_dict[cam_name] for cam_name in self.camera_names
                ]
                all_cam_images = np.stack(all_cam_images, axis=0)
                image_sequence.append(all_cam_images)

            image_sequence = np.array(image_sequence)
            image_sequence = torch.tensor(image_sequence, dtype=torch.float32)
            image_sequence = torch.einsum("t k h w c -> t k c h w", image_sequence)
            image_sequence = image_sequence / 255.0

        return image_sequence, command_embedding, command_gt


def load_merged_data(
    dataset_dirs,
    num_episodes_list,
    camera_names,
    batch_size_train,
    batch_size_val,
    history_len=1,
    prediction_offset=10,
    history_skip_frame=1,
    random_crop=False,
    dagger_ratio=None,
    use_augmentation=False,
    image_size=224,
):
    assert len(dataset_dirs) == len(
        num_episodes_list
    ), "Length of dataset_dirs and num_episodes_list must be the same."
    print(f"{history_len=}, {history_skip_frame=}, {prediction_offset=}")
    if random_crop:
        print(f"Random crop enabled")
    if use_augmentation:
        print(f"Data augmentation enabled")
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1."

    all_episode_indices = []
    last_dataset_indices = []

    for i, (dataset_dir, num_episodes) in enumerate(
        zip(dataset_dirs, num_episodes_list)
    ):
        print(f"\nData from: {dataset_dir}\n")

        # Get episode indices for current dataset
        episode_indices = [(dataset_dir, i) for i in range(num_episodes)]
        if i == len(dataset_dirs) - 1:  # Last dataset
            last_dataset_indices.extend(episode_indices)
        all_episode_indices.extend(episode_indices)

    print(f"Total number of episodes across datasets: {len(all_episode_indices)}")

    # Obtain train test split
    train_ratio = 0.95
    shuffled_indices = np.random.permutation(all_episode_indices)
    train_indices = shuffled_indices[: int(train_ratio * len(all_episode_indices))]
    val_indices = shuffled_indices[int(train_ratio * len(all_episode_indices)) :]

    # Construct dataset and dataloader for each dataset dir and merge them
    train_datasets = [
        SequenceDataset(
            [idx for d, idx in train_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            history_len,
            prediction_offset,
            history_skip_frame,
            random_crop,
            use_augmentation=use_augmentation,
            training=True,
            image_size=image_size,
        )
        for dataset_dir in dataset_dirs
    ]
    val_datasets = [
        SequenceDataset(
            [idx for d, idx in val_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            history_len,
            prediction_offset,
            history_skip_frame,
            random_crop,
            use_augmentation=use_augmentation,
            training=False,
            image_size=image_size,
        )
        for dataset_dir in dataset_dirs
    ]
    all_datasets = [
        SequenceDataset(
            [idx for d, idx in all_episode_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            history_len,
            prediction_offset,
            history_skip_frame,
            random_crop,
            use_augmentation=use_augmentation,
            training=False,  # Test mode, no augmentation
            image_size=image_size,
        )
        for dataset_dir in dataset_dirs
    ]

    merged_train_dataset = ConcatDataset(train_datasets)
    merged_val_dataset = ConcatDataset(val_datasets)
    merged_all_dataset = ConcatDataset(all_datasets)

    if dagger_ratio is not None:  # Use all data. TODO: add val_dataloader
        dataset_sizes = {
            dataset_dir: num_episodes
            for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        }
        dagger_sampler = DAggerSampler(
            all_episode_indices,
            last_dataset_indices,
            batch_size_train,
            dagger_ratio,
            dataset_sizes,
        )
        train_dataloader = DataLoader(
            merged_all_dataset,
            batch_sampler=dagger_sampler,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )
        val_dataloader = None
    else:
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
        val_dataloader = DataLoader(
            merged_val_dataset,
            batch_size=batch_size_val,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
    test_dataloader = DataLoader(
        merged_all_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=20,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, test_dataloader


# ===================== SPLITTED DATASETS ADAPTER =====================

class SplittedSequenceDataset(torch.utils.data.Dataset):
    """
    Dataset adapter for splitted dataset format (stage*/run_*).
    Adapted from act.utils.SplittedEpisodicDataset for HighLevelModel training.
    """
    def __init__(
        self,
        root_dir: str,
        camera_names,
        history_len=5,
        prediction_offset=10,
        history_skip_frame=1,
        random_crop=False,
        stage_embeddings: dict | None = None,
        stage_texts: dict | None = None,
        use_augmentation=False,
        training=True,
        image_size=224,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.camera_names = camera_names
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.random_crop = random_crop
        self.stage_embeddings = stage_embeddings or {}
        self.stage_texts = stage_texts or {}
        self.use_augmentation = use_augmentation
        self.training = training
        self.image_size = image_size
        
        # Initialize ImagePreprocessor if augmentation is enabled
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation
            )
        
        # Find all runs
        self.runs = []  # list of (stage_idx, run_path)
        for stage_dir in sorted(os.listdir(root_dir)):
            stage_path = os.path.join(root_dir, stage_dir)
            if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                continue
            try:
                stage_idx = int(stage_dir[5:]) - 1  # stage1 -> 0, stage2 -> 1, etc.
            except Exception:
                continue
            for run_dir in sorted(os.listdir(stage_path)):
                run_path = os.path.join(stage_path, run_dir)
                if run_dir.startswith("run") and os.path.isdir(run_path):
                    self.runs.append((stage_idx, run_path))
        
        if len(self.runs) == 0:
            print(f"[SplittedSequenceDataset] No runs found under {root_dir}")
        
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.runs)
    
    def _parse_run(self, run_path: str):
        """Parse data.txt to extract labels (stages), actions, and qpos."""
        data_file = os.path.join(run_path, 'data.txt')
        if not os.path.exists(data_file):
            return None, None, None
        
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        labels, actions, qpos = [], [], []
        frame_blocks = re.split(r'Frame_\d+:', content)[1:]
        
        for b in frame_blocks:
            m_stage = re.search(r'stage\s*:\s*(\d+)', b)
            m_act = re.search(r'action:\s*\[([^\]]+)\]', b)
            m_lr = re.search(r'lrstate\s*:\s*\[([^\]]+)\]', b)
            if m_stage and m_act and m_lr:
                try:
                    labels.append(int(m_stage.group(1)) - 1)  # stage1 -> 0
                    actions.append(np.array([float(x.strip()) for x in m_act.group(1).split(',')], dtype=np.float32))
                    qpos.append(np.array([float(x.strip()) for x in m_lr.group(1).split(',')], dtype=np.float32))
                except Exception:
                    continue
        
        if not actions:
            return None, None, None
        
        return labels, np.stack(actions), np.stack(qpos)
    
    def __getitem__(self, index):
        stage_idx, run_path = self.runs[index]
        
        # Parse run data
        labels, actions_np, qpos_np = self._parse_run(run_path)
        
        if actions_np is None or len(actions_np) < 2:
            # Fallback: return a different sample
            return self.__getitem__((index + 1) % len(self.runs))
        
        total_timesteps = len(actions_np)
        
        # Sample a random curr_ts and compute start_ts and target_ts
        try:
            min_start = self.history_len * self.history_skip_frame
            max_start = total_timesteps - self.prediction_offset - 1
            if max_start < min_start:
                # Not enough timesteps, try another sample
                return self.__getitem__((index + 1) % len(self.runs))
            
            curr_ts = np.random.randint(min_start, max_start + 1)
            start_ts = curr_ts - self.history_len * self.history_skip_frame
            target_ts = curr_ts + self.prediction_offset
        except ValueError:
            return self.__getitem__((index + 1) % len(self.runs))
        
        # Get command embedding and text for target_ts
        # Use the stage at target_ts (or closest available)
        target_stage_idx = labels[min(target_ts, len(labels) - 1)] if labels else stage_idx
        
        command_embedding = self.stage_embeddings.get(target_stage_idx, None)
        if command_embedding is None:
            command_embedding = torch.zeros(768)
        else:
            command_embedding = torch.tensor(command_embedding).float().squeeze()
        
        command_gt = self.stage_texts.get(target_stage_idx, f"stage_{target_stage_idx + 1}")
        
        # Load image sequence for history frames
        image_sequence = []
        for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
            image_dict = {}
            for cam_name in self.camera_names:
                cam_dir = os.path.join(run_path, cam_name)
                if not os.path.exists(cam_dir):
                    # Fallback: try another sample
                    return self.__getitem__((index + 1) % len(self.runs))
                
                files = sorted([f for f in os.listdir(cam_dir) if f.endswith('_full.jpg')])
                if not files:
                    return self.__getitem__((index + 1) % len(self.runs))
                
                # Use frame index corresponding to ts
                file_idx = min(ts, len(files) - 1)
                img_path = os.path.join(cam_dir, files[file_idx])
                
                # Load and convert image
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                
                # Apply transformations similar to original SequenceDataset
                if cam_name == "cam_high":
                    img_np = crop_resize(img_np)
                if self.random_crop:
                    img_np = random_crop(img_np)
                
                # Apply data augmentation if enabled
                if self.use_augmentation:
                    img_np = self.image_preprocessor.augment_image(
                        img_np, self.training
                    )
                
                # Convert to tensor format (C, H, W) in [0, 1]
                img_t = self.to_tensor(img_np)
                image_dict[cam_name] = img_t
            
            # Stack all camera images: (num_cameras, C, H, W)
            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images = torch.stack(all_cam_images, dim=0)
            image_sequence.append(all_cam_images)
        
        # Stack sequence: (timesteps, num_cameras, C, H, W)
        image_sequence = torch.stack(image_sequence, dim=0)
        
        return image_sequence, command_embedding, command_gt


def load_splitted_data(
    root_dir: str,
    camera_names,
    batch_size_train: int,
    batch_size_val: int,
    history_len=1,
    prediction_offset=10,
    history_skip_frame=1,
    random_crop=False,
    stage_embeddings_file: str | None = None,
    stage_texts_file: str | None = None,
    dagger_ratio=None,
    train_subdir: str = "training_datasets",
    val_subdir: str = "validation_datasets",
    use_augmentation=False,
    image_size=224,
):
    """
    Load splitted dataset for HighLevelModel training.
    
    Args:
        root_dir: Root directory containing training_datasets/ and validation_datasets/ subdirectories
                  (or directly containing stage*/run_* folders if train_subdir/val_subdir are None)
        camera_names: List of camera names
        batch_size_train: Training batch size
        batch_size_val: Validation batch size
        history_len: Number of history frames
        prediction_offset: Offset for prediction target
        history_skip_frame: Frame skip for history
        random_crop: Whether to apply random crop
        stage_embeddings_file: JSON file with stage embeddings
        stage_texts_file: JSON file with stage texts (optional, for command text)
        dagger_ratio: DAgger ratio (optional)
        train_subdir: Subdirectory name for training data (default: "training_datasets")
        val_subdir: Subdirectory name for validation data (default: "validation_datasets")
        use_augmentation: Whether to use data augmentation (default: False)
        image_size: Image size for augmentation (default: 224)
    """
    # Load stage embeddings
    stage_embeddings = {}
    if stage_embeddings_file and os.path.exists(stage_embeddings_file):
        with open(stage_embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get('stage_embeddings', data)
        if isinstance(entries, list):
            stage_embeddings = {int(e['stage']): e['embedding'] for e in entries if 'stage' in e and 'embedding' in e}
        elif isinstance(entries, dict):
            # Only convert keys that can be converted to int (stage numbers)
            stage_embeddings = {}
            for k, v in entries.items():
                try:
                    stage_idx = int(k)
                    stage_embeddings[stage_idx] = v
                except (ValueError, TypeError):
                    # Skip non-numeric keys (like 'model', 'encoder', etc.)
                    continue
    
    # Load stage texts (optional)
    stage_texts = {}
    if stage_texts_file and os.path.exists(stage_texts_file):
        with open(stage_texts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get('stage_texts', data)
        if isinstance(entries, list):
            stage_texts = {int(e['stage']): e['text'] for e in entries if 'stage' in e and 'text' in e}
        elif isinstance(entries, dict):
            # Only convert keys that can be converted to int (stage numbers)
            stage_texts = {}
            for k, v in entries.items():
                try:
                    stage_idx = int(k)
                    stage_texts[stage_idx] = v
                except (ValueError, TypeError):
                    # Skip non-numeric keys (like 'model', 'encoder', etc.)
                    continue
    
    # Determine train and validation root directories
    train_root_dir = os.path.join(root_dir, train_subdir) if train_subdir else root_dir
    val_root_dir = os.path.join(root_dir, val_subdir) if val_subdir else None
    
    # Check if training directory exists
    if not os.path.exists(train_root_dir):
        # Fallback: try root_dir directly
        if os.path.exists(root_dir) and any(d.startswith("stage") for d in os.listdir(root_dir)):
            train_root_dir = root_dir
            val_root_dir = None
        else:
            raise ValueError(f"Training directory not found: {train_root_dir}")
    
    if use_augmentation:
        print(f"Data augmentation enabled")
    
    # Create training dataset
    train_dataset = SplittedSequenceDataset(
        train_root_dir,
        camera_names,
        history_len=history_len,
        prediction_offset=prediction_offset,
        history_skip_frame=history_skip_frame,
        random_crop=random_crop,
        stage_embeddings=stage_embeddings,
        stage_texts=stage_texts,
        use_augmentation=use_augmentation,
        training=True,
        image_size=image_size,
    )
    
    print(f"Training dataset: {len(train_dataset)} samples from {train_root_dir}")
    
    # Create validation dataset if validation directory exists
    val_dataset = None
    if val_root_dir and os.path.exists(val_root_dir):
        val_dataset = SplittedSequenceDataset(
            val_root_dir,
            camera_names,
            history_len=history_len,
            prediction_offset=prediction_offset,
            history_skip_frame=history_skip_frame,
            random_crop=random_crop,
            stage_embeddings=stage_embeddings,
            stage_texts=stage_texts,
            use_augmentation=use_augmentation,
            training=False,
            image_size=image_size,
        )
        print(f"Validation dataset: {len(val_dataset)} samples from {val_root_dir}")
    else:
        # If no separate validation directory, split training data
        if val_root_dir:
            print(f"Warning: Validation directory {val_root_dir} not found, splitting training data")
        total_size = len(train_dataset)
        train_ratio = 0.95
        val_ratio = 0.05
        test_ratio = 0.0
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        )
        print(f"Split training data: {train_size} train, {val_size} val, {test_size} test")
    
    # Create test dataset (use validation data if available, otherwise split from train)
    if val_root_dir and os.path.exists(val_root_dir):
        # Use validation data as test data
        test_dataset = val_dataset
    elif 'test_dataset' not in locals():
        # If no validation directory and test_dataset wasn't created, use train_dataset
        test_dataset = train_dataset
    
    # Create dataloaders
    if dagger_ratio is not None:
        # For DAgger, use all training data
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=8,
            persistent_workers=True,
        )
        val_dataloader = None
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=16,
            persistent_workers=True,
        )
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=True,
                pin_memory=True,
                num_workers=8,
                prefetch_factor=16,
                persistent_workers=True,
            )
        else:
            val_dataloader = None
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=20,
        prefetch_factor=1,
    )
    
    return train_dataloader, val_dataloader, test_dataloader


"""
Test the SequenceDataset class.

Example usage:
$ python src/HighLevelModel/dataset.py --dataset_dir /scr/lucyshi/dataset/aloha_bag_3_objects
"""
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()

    # Parameters for the test
    camera_names = ["cam_high", "cam_low"]
    history_len = 5
    prediction_offset = 10
    num_episodes = 10  # Just to sample from the first 10 episodes for testing

    # Create a SequenceDataset instance
    dataset = SequenceDataset(
        list(range(num_episodes)),
        args.dataset_dir,
        camera_names,
        history_len,
        prediction_offset,
    )

    # Sample a random item from the dataset
    idx = np.random.randint(0, len(dataset))
    image_sequence, command_embedding, _ = dataset[idx]

    print(f"Sampled episode index: {idx}")
    print(f"Image sequence shape: {image_sequence.shape}")
    print(f"Language embedding shape: {command_embedding.shape}")

    # Save the images in the sequence
    for t in range(history_len):
        plt.figure(figsize=(10, 5))
        for cam_idx, cam_name in enumerate(camera_names):
            plt.subplot(1, len(camera_names), cam_idx + 1)
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(
                image_sequence[t, cam_idx].permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB
            )
            plt.imshow(img_rgb)
            plt.title(f"{cam_name} at timestep {t}")
        plt.tight_layout()
        plt.savefig(f"plot/image_sequence_timestep_{t}.png")
        print(f"Saved image_sequence_timestep_{t}.png")
