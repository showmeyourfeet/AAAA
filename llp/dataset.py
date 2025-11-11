"""Dataset classes and data loading utilities for ACT."""

import json
import os
import random
import re
from typing import Dict, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torchvision import transforms
import torchvision.transforms as T

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None

from .utils import crop_resize


CROP_TOP = True
FILTER_MISTAKES = True


class ImagePreprocessor:
    """Image preprocessor with data augmentation support.
    
    Based on the ImagePreprocessor from HighLevelModel/dataset.py, adapted for ACT training.
    """
    def __init__(self, image_size: int = 224, use_augmentation: bool = True, normalize: bool = False):
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.normalize = normalize

        # Define Albumentations enhancement process
        if use_augmentation and ALBUMENTATIONS_AVAILABLE:
            self.albumentations_transform = A.Compose([
                # Use multiple enhancement methods
                A.OneOf([
                    A.Rotate(limit=[-10, 10], p=0.5),
                    A.Affine(rotate=[-10, 10],scale=[0.9, 1.1], translate_percent=[-0.1, 0.1], shear=[-10, 10], p=0.8),
                ], p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
                ], p=0.5),
                A.CoarseDropout(num_holes_range=[1, 3], hole_height_range=[0.1, 0.2], hole_width_range=[0.1, 0.2], p=0.5)
            ])
        else:
            # Evaluation: no augmentation or albumentations not available
            self.albumentations_transform = None
            if use_augmentation and not ALBUMENTATIONS_AVAILABLE:
                print("Warning: albumentations not available, data augmentation disabled. Install with: pip install albumentations")

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


class EpisodicDataset(torch.utils.data.Dataset):
    """Dataset for loading HDF5 episodes with optional language filtering."""

    def __init__(
        self,
        episode_ids: Sequence[int],
        dataset_dir: str,
        camera_names: Sequence[str],
        norm_stats: Dict[str, np.ndarray],
        max_len: int | None = None,
        command_list: Sequence[str] | None = None,
        use_language: bool = False,
        language_encoder: str | None = None,
        policy_class: str | None = None,
        use_augmentation: bool = False,
        image_size: int = 224,
        training: bool = True,
    ) -> None:
        super().__init__()
        episode_ids = list(episode_ids)
        self.episode_ids = episode_ids if len(episode_ids) > 0 else [0]
        self.dataset_dir = dataset_dir
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.is_sim: bool | None = None
        self.max_len = max_len
        self.command_list = [cmd.strip("'\"") for cmd in (command_list or [])]
        self.use_language = use_language
        self.language_encoder = language_encoder
        self.policy_class = policy_class
        self.use_augmentation = use_augmentation
        self.image_size = image_size
        self.training = training
        self.transformations = None

        # Initialize ImagePreprocessor if augmentation is enabled
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation
            )

        # Initialize self.is_sim
        self.__getitem__(0)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int):
        max_len = self.max_len
        assert max_len is not None, "max_len must be provided"

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")

        if self.use_language or FILTER_MISTAKES:
            assert self.language_encoder is not None, "language_encoder is required"
            json_name = f"episode_{episode_id}_encoded_{self.language_encoder}.json"
            encoded_json_path = os.path.join(self.dataset_dir, json_name)
            with open(encoded_json_path, "r", encoding="utf-8") as f:
                episode_data = json.load(f)
        else:
            episode_data = []

        if len(self.command_list) > 0:
            matching_segments = []
            for segment in episode_data:
                if segment["command"] in self.command_list:
                    current_idx = episode_data.index(segment)
                    if (
                        current_idx + 1 < len(episode_data)
                        and episode_data[current_idx + 1]["type"] == "correction"
                    ):
                        continue
                    matching_segments.append(segment)

            if not matching_segments:
                raise ValueError(
                    f"No matching segments found for episode {episode_id}"
                )

            chosen_segment = random.choice(matching_segments)
            segment_start = chosen_segment["start_timestep"]
            segment_end = chosen_segment["end_timestep"]
            if segment_start is None or segment_end is None:
                raise ValueError(
                    f"Command segment not found for episode {episode_id}"
                )
            command_embedding = (
                torch.tensor(chosen_segment["embedding"]).squeeze()
                if self.use_language
                else None
            )
        elif self.use_language or FILTER_MISTAKES:
            while True:
                segment = random.choice(episode_data)
                current_idx = episode_data.index(segment)
                if (
                    current_idx + 1 < len(episode_data)
                    and episode_data[current_idx + 1]["type"] == "correction"
                ):
                    continue
                segment_start = segment["start_timestep"]
                segment_end = segment["end_timestep"]
                if segment_end - segment_start + 1 < 20:
                    continue
                command_embedding = torch.tensor(segment["embedding"]).squeeze()
                break
        else:
            segment_start = segment_end = None
            command_embedding = None

        with h5py.File(dataset_path, "r") as root:
            is_sim = bool(root.attrs.get("sim", False))
            self.is_sim = is_sim
            compressed = root.attrs.get("compress", False)
            original_action_shape = root["/action"].shape

            if len(self.command_list) > 0 or self.use_language:
                assert segment_start is not None and segment_end is not None
                start_ts = np.random.randint(segment_start, segment_end)
                end_ts = min(segment_end, start_ts + max_len - 2)
            else:
                start_ts = np.random.choice(original_action_shape[0])
                end_ts = original_action_shape[0] - 1

            qpos = root["/observations/qpos"][start_ts]

            image_dict = {}
            for cam_name in self.camera_names:
                image = root[f"/observations/images/{cam_name}"][start_ts]
                if compressed:
                    image = cv2.imdecode(image, 1)
                if CROP_TOP and cam_name == "cam_high":
                    image = crop_resize(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply data augmentation if enabled
                if self.use_augmentation:
                    image = self.image_preprocessor.augment_image(image, self.training)
                
                image_dict[cam_name] = image

            all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
            all_cam_images = np.stack(all_cam_images, axis=0)
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum("k h w c -> k c h w", image_data)

            if self.transformations is None:
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(
                        size=[
                            int(original_size[0] * ratio),
                            int(original_size[1] * ratio),
                        ]
                    ),
                    transforms.Resize(original_size, antialias=True),
                ]
                if self.policy_class == "Diffusion":
                    self.transformations.extend(
                        [
                            transforms.RandomRotation(
                                degrees=[-5.0, 5.0], expand=False
                            ),
                            transforms.ColorJitter(
                                brightness=0.3, contrast=0.4, saturation=0.5
                            ),
                        ]
                    )

            for transform in self.transformations:
                image_data = transform(image_data)

            image_data = image_data / 255.0

            if is_sim:
                action = root["/action"][start_ts : end_ts + 1]
                action_len = end_ts - start_ts + 1
            else:
                action = root["/action"][max(0, start_ts - 1) : end_ts + 1]
                action_len = end_ts - max(0, start_ts - 1) + 1

            padded_action = np.zeros(
                (max_len,) + original_action_shape[1:], dtype=np.float32
            )
            padded_action[:action_len] = action
            is_pad = np.zeros(max_len)
            is_pad[action_len:] = 1

            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()
            qpos_data = torch.from_numpy(qpos).float()

            if self.policy_class == "Diffusion":
                action_data = (
                    (action_data - self.norm_stats["action_min"])
                    / (
                        self.norm_stats["action_max"]
                        - self.norm_stats["action_min"]
                    )
                ) * 2 - 1
            else:
                action_data = (
                    action_data - self.norm_stats["action_mean"]
                ) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
                "qpos_std"
            ]

            if self.use_language:
                assert command_embedding is not None
                return image_data, qpos_data, action_data, is_pad, command_embedding
            return image_data, qpos_data, action_data, is_pad


def get_norm_stats(
    dataset_dirs: Sequence[str], num_episodes_list: Sequence[int]
) -> Dict[str, np.ndarray]:
    all_qpos_data = []
    all_action_data = []

    for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list):
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
            with h5py.File(dataset_path, "r") as root:
                qpos = root["/observations/qpos"][()]
                action = root["/action"][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()
    eps = 1e-4

    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "action_min": action_min.numpy() - eps,
        "action_max": action_max.numpy() + eps,
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": all_qpos_data[-1].numpy(),
    }
    return stats


class SplittedEpisodicDataset(torch.utils.data.Dataset):
    """Dataset adapter for stage/run-style datasets."""

    def __init__(
        self,
        root_dir: str,
        camera_names: Sequence[str],
        norm_stats: Dict[str, np.ndarray],
        max_len: int,
        use_language: bool = False,
        stage_embeddings: Dict[int, Sequence[float]] | None = None,
        image_size: int = 224,
        use_augmentation: bool = False,
        training: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.max_len = max_len
        self.use_language = use_language
        self.stage_embeddings = stage_embeddings or {}
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.training = training
        
        # Initialize ImagePreprocessor if augmentation is enabled
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation
            )

        self.runs: list[Tuple[int, str]] = []
        for stage_dir in sorted(os.listdir(root_dir)):
            stage_path = os.path.join(root_dir, stage_dir)
            if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                continue
            try:
                stage_idx = int(stage_dir[5:]) - 1
            except ValueError:
                continue
            for run_dir in sorted(os.listdir(stage_path)):
                run_path = os.path.join(stage_path, run_dir)
                if run_dir.startswith("run") and os.path.isdir(run_path):
                    self.runs.append((stage_idx, run_path))

        if len(self.runs) == 0:
            print(f"[SplittedEpisodicDataset] No runs found under {root_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.runs)

    def _parse_run(self, run_path: str):
        data_file = os.path.join(run_path, "data.txt")
        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()

        labels: list[int] = []
        actions: list[np.ndarray] = []
        qpos: list[np.ndarray] = []

        frame_blocks = re.split(r"Frame_\d+:", content)[1:]
        for block in frame_blocks:
            m_stage = re.search(r"stage\s*:\s*(\d+)", block)
            m_act = re.search(r"action:\s*\[([^\]]+)\]", block)
            m_lr = re.search(r"lrstate\s*:\s*\[([^\]]+)\]", block)
            if m_stage and m_act and m_lr:
                try:
                    labels.append(int(m_stage.group(1)) - 1)
                    actions.append(
                        np.array(
                            [
                                float(x.strip())
                                for x in m_act.group(1).split(",")
                            ],
                            dtype=np.float32,
                        )
                    )
                    qpos.append(
                        np.array(
                            [
                                float(x.strip())
                                for x in m_lr.group(1).split(",")
                            ],
                            dtype=np.float32,
                        )
                    )
                except Exception:
                    continue
        return labels, np.stack(actions) if actions else None, np.stack(qpos) if qpos else None

    def __getitem__(self, index: int):
        stage_idx, run_path = self.runs[index]
        labels, actions_np, qpos_np = self._parse_run(run_path)

        if actions_np is None or len(actions_np) < 2:
            actions_np = np.zeros(
                (self.max_len, self.norm_stats["action_mean"].shape[0]),
                dtype=np.float32,
            )
            qpos_np = np.zeros(
                (self.max_len, self.norm_stats["qpos_mean"].shape[0]),
                dtype=np.float32,
            )
            start_ts, end_ts = 0, min(self.max_len - 1, actions_np.shape[0] - 1)
        else:
            T = actions_np.shape[0]
            start_ts = np.random.randint(0, max(1, T - 1))
            end_ts = min(T - 1, start_ts + self.max_len - 1)

        qpos = qpos_np[start_ts]

        all_cam_images = []
        for cam in self.camera_names:
            cam_dir = os.path.join(run_path, cam)
            files = sorted(
                [f for f in os.listdir(cam_dir) if f.endswith("_full.jpg")]
            )
            if not files:
                raise FileNotFoundError(
                    f"No *_full.jpg files found under {cam_dir}"
                )
            idx = min(start_ts, len(files) - 1)
            img = Image.open(os.path.join(cam_dir, files[idx])).convert("RGB")
            
            # Apply data augmentation if enabled
            if self.use_augmentation:
                img_np = np.array(img)
                img_np = self.image_preprocessor.augment_image(img_np, self.training)
                img = Image.fromarray(img_np)
            
            img_t = self.to_tensor(img)
            all_cam_images.append(img_t)

        image_data = torch.stack(all_cam_images, dim=0)

        action_slice = actions_np[start_ts : end_ts + 1]
        action_len = action_slice.shape[0]
        padded_action = np.zeros(
            (self.max_len, action_slice.shape[1]), dtype=np.float32
        )
        padded_action[:action_len] = action_slice
        is_pad = np.zeros(self.max_len, dtype=np.bool_)
        is_pad[action_len:] = True

        action_data = torch.from_numpy(padded_action).float()
        action_data = (
            action_data - torch.from_numpy(self.norm_stats["action_mean"]).float()
        ) / torch.from_numpy(self.norm_stats["action_std"]).float()

        qpos_data = torch.from_numpy(qpos).float()
        qpos_data = (
            qpos_data - torch.from_numpy(self.norm_stats["qpos_mean"]).float()
        ) / torch.from_numpy(self.norm_stats["qpos_std"]).float()
        is_pad = torch.from_numpy(is_pad).bool()

        if self.use_language:
            emb = self.stage_embeddings.get(stage_idx, None)
            if emb is None:
                emb = torch.zeros(768)
            else:
                emb = torch.tensor(emb).float()
            return image_data, qpos_data, action_data, is_pad, emb
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats_splitted(root_dir: str) -> Dict[str, np.ndarray]:
    all_qpos, all_actions = [], []
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
            f"No valid actions/qpos to compute stats under {root_dir}"
        )

    all_actions = torch.cat(all_actions, dim=0)
    all_qpos = torch.cat(all_qpos, dim=0)
    action_mean = all_actions.mean(dim=0).float().numpy()
    action_std = torch.clip(all_actions.std(dim=0), 1e-2).float().numpy()
    qpos_mean = all_qpos.mean(dim=0).float().numpy()
    qpos_std = torch.clip(all_qpos.std(dim=0), 1e-2).float().numpy()
    return {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
    }


def load_splitted_data(
    root_dir: str,
    camera_names: Sequence[str],
    batch_size_train: int,
    max_len: int,
    use_language: bool = False,
    stage_embeddings_file: str | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
):
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

    norm_stats = get_norm_stats_splitted(root_dir)
    dataset = SplittedEpisodicDataset(
        root_dir,
        camera_names,
        norm_stats,
        max_len=max_len,
        use_language=use_language,
        stage_embeddings=stage_embeddings,
        image_size=image_size,
        use_augmentation=use_augmentation,
        training=True,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=2,
        persistent_workers=False,
    )
    return train_dataloader, norm_stats, False


def load_merged_data(
    dataset_dirs: Sequence[str],
    num_episodes_list: Sequence[int],
    camera_names: Sequence[str],
    batch_size_train: int,
    max_len: int | None = None,
    command_list: Sequence[str] | None = None,
    use_language: bool = False,
    language_encoder: str | None = None,
    dagger_ratio: float | None = None,
    policy_class: str | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
):
    if dagger_ratio is not None:
        assert 0 <= dagger_ratio <= 1, "dagger_ratio must be between 0 and 1"

    all_filtered_indices: list[Tuple[str, int]] = []
    last_dataset_indices: list[Tuple[str, int]] = []

    for i, (dataset_dir, num_episodes) in enumerate(
        zip(dataset_dirs, num_episodes_list)
    ):
        print(f"\nData from: {dataset_dir}\n")

        if command_list:
            cleaned_commands = [cmd.strip("'\"") for cmd in command_list]
            filtered_indices = []
            for episode_id in range(num_episodes):
                json_path = os.path.join(dataset_dir, f"episode_{episode_id}.json")
                with open(json_path, "r", encoding="utf-8") as f:
                    instruction_data = json.load(f)
                for segment in instruction_data:
                    if segment["command"] in cleaned_commands:
                        current_idx = instruction_data.index(segment)
                        if (
                            current_idx + 1 < len(instruction_data)
                            and instruction_data[current_idx + 1]["type"]
                            == "correction"
                        ):
                            continue
                        filtered_indices.append((dataset_dir, episode_id))
                        break
        else:
            filtered_indices = [(dataset_dir, j) for j in range(num_episodes)]

        if i == len(dataset_dirs) - 1:
            last_dataset_indices.extend(filtered_indices)
        all_filtered_indices.extend(filtered_indices)

    print(
        f"Total number of episodes across datasets: {len(all_filtered_indices)}"
    )

    norm_stats = get_norm_stats(dataset_dirs, num_episodes_list)

    train_datasets = [
        EpisodicDataset(
            [idx for d, idx in all_filtered_indices if d == dataset_dir],
            dataset_dir,
            camera_names,
            norm_stats,
            max_len,
            command_list or [],
            use_language,
            language_encoder,
            policy_class,
            use_augmentation=use_augmentation,
            image_size=image_size,
            training=True,
        )
        for dataset_dir in dataset_dirs
    ]
    merged_train_dataset = ConcatDataset(train_datasets)

    if dagger_ratio is not None:
        dataset_sizes = {
            dataset_dir: num_episodes
            for dataset_dir, num_episodes in zip(dataset_dirs, num_episodes_list)
        }
        dagger_sampler = DAggerSampler(
            all_filtered_indices,
            last_dataset_indices,
            batch_size_train,
            dagger_ratio,
            dataset_sizes,
        )
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_sampler=dagger_sampler,
            pin_memory=True,
            num_workers=24,
            prefetch_factor=4,
            persistent_workers=True,
        )
    else:
        train_dataloader = DataLoader(
            merged_train_dataset,
            batch_size=batch_size_train,
            shuffle=True,
            pin_memory=True,
            num_workers=24,
            prefetch_factor=4,
            persistent_workers=True,
        )

    return train_dataloader, norm_stats, train_datasets[-1].is_sim


class DAggerSampler(Sampler[list[int]]):
    def __init__(
        self,
        all_indices: Sequence[Tuple[str, int]],
        last_dataset_indices: Sequence[Tuple[str, int]],
        batch_size: int,
        dagger_ratio: float,
        dataset_sizes: Dict[str, int],
    ) -> None:
        self.other_indices, self.last_dataset_indices = self._flatten_indices(
            all_indices, last_dataset_indices, dataset_sizes
        )
        print(
            "Len of data from the last dataset:"
            f" {len(self.last_dataset_indices)}, Len of data from other datasets:"
            f" {len(self.other_indices)}"
        )
        self.batch_size = batch_size
        self.dagger_ratio = dagger_ratio
        self.num_batches = len(all_indices) // self.batch_size

    @staticmethod
    def _flatten_indices(
        all_indices: Sequence[Tuple[str, int]],
        last_dataset_indices: Sequence[Tuple[str, int]],
        dataset_sizes: Dict[str, int],
    ) -> Tuple[list[int], list[int]]:
        flat_other_indices: list[int] = []
        flat_last_dataset_indices: list[int] = []
        cumulative_size = 0

        for dataset_dir, size in dataset_sizes.items():
            for idx in range(size):
                if (dataset_dir, idx) in last_dataset_indices:
                    flat_last_dataset_indices.append(cumulative_size + idx)
                elif (dataset_dir, idx) in all_indices:
                    flat_other_indices.append(cumulative_size + idx)
            cumulative_size += size

        return flat_other_indices, flat_last_dataset_indices

    def __iter__(self):
        num_samples_last = int(self.batch_size * self.dagger_ratio)
        num_samples_other = self.batch_size - num_samples_last

        for _ in range(self.num_batches):
            batch_indices: list[int] = []

            if num_samples_last > 0 and self.last_dataset_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.last_dataset_indices, num_samples_last, replace=True
                    )
                )

            if num_samples_other > 0 and self.other_indices:
                batch_indices.extend(
                    np.random.choice(
                        self.other_indices, num_samples_other, replace=True
                    )
                )

            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches


# =========================
# Paired adjacent-stage dataset (splitted)
# =========================

def _list_stage_runs(root_dir: str) -> dict[int, list[str]]:
    """Scan splitted root to build {stage_index_1based: [run_paths...]}."""
    stage_to_runs: dict[int, list[str]] = {}
    for entry in sorted(os.listdir(root_dir)):
        stage_path = os.path.join(root_dir, entry)
        if not (entry.startswith("stage") and os.path.isdir(stage_path)):
            continue
        try:
            stage_idx = int(entry[5:])  # 'stage12' -> 12
        except ValueError:
            continue
        runs: list[str] = []
        for run_dir in sorted(os.listdir(stage_path)):
            run_path = os.path.join(stage_path, run_dir)
            if run_dir.startswith("run") and os.path.isdir(run_path):
                runs.append(run_path)
        if runs:
            stage_to_runs[stage_idx] = runs
    if not stage_to_runs:
        raise RuntimeError(f"No stage*/run_* found under {root_dir}")
    return stage_to_runs


def _count_frames_in_run(run_path: str, camera_names: Sequence[str]) -> int:
    """Count available frames for a run by inspecting the first camera."""
    if len(camera_names) == 0:
        raise ValueError("camera_names must be non-empty")
    cam_dir = os.path.join(run_path, camera_names[0])
    files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_full.jpg")])
    if not files:
        raise RuntimeError(f"No *_full.jpg under {cam_dir}")
    return len(files)


def _collect_stage_embeddings(stage_embeddings_file: str | None) -> dict[int, list[float]]:
    """Load stage embeddings json into {stage_1based: embedding_list}."""
    if not stage_embeddings_file:
        return {}
    if not os.path.exists(stage_embeddings_file):
        raise FileNotFoundError(f"Stage embeddings not found: {stage_embeddings_file}")
    with open(stage_embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("stage_embeddings", data)
    out: dict[int, list[float]] = {}
    if isinstance(entries, list):
        for e in entries:
            if "stage" in e and "embedding" in e:
                try:
                    out[int(e["stage"])] = e["embedding"]
                except Exception:
                    continue
    elif isinstance(entries, dict):
        for k, v in entries.items():
            try:
                out[int(k)] = v
            except Exception:
                continue
    return out

def _collect_stage_texts(stage_texts_file: str | None) -> dict[int, str]:
    """Load stage texts from json. Supports either {'stage_texts': {...}} or 'stage_embeddings' entries with 'text'."""
    if not stage_texts_file:
        return {}
    if not os.path.exists(stage_texts_file):
        return {}
    with open(stage_texts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Direct dict form
    texts_obj = data.get("stage_texts", None)
    stage_texts: dict[int, str] = {}
    if isinstance(texts_obj, dict):
        for k, v in texts_obj.items():
            try:
                stage_texts[int(k)] = str(v)
            except Exception:
                continue
    # Fallback: extract from stage_embeddings list/dict
    entries = data.get("stage_embeddings", data)
    if isinstance(entries, list):
        for e in entries:
            if "stage" in e and "text" in e:
                try:
                    stage_texts[int(e["stage"])] = str(e["text"])
                except Exception:
                    continue
    elif isinstance(entries, dict):
        # entries like {"1": {"text": "...", "embedding": [...]}, ...} or {"1": [..]} (no text)
        for k, v in entries.items():
            if isinstance(v, dict) and "text" in v:
                try:
                    stage_texts[int(k)] = str(v["text"])
                except Exception:
                    continue
    return stage_texts


class PairedStageSequenceDataset(torch.utils.data.Dataset):
    """
    Construct paired sequences by concatenating runs from adjacent stages (i, i+1).
    Sampling keeps: random current_frame, build history with skip_frame, and target at offset.

    Each item returns:
      - image_sequence: (T, K, C, H, W) float tensor in [0,1]
      - command_embedding: (768,) float tensor (zero if missing)
      - command_text: str like 'stage_{j}'
    """

    def __init__(
        self,
        pairs: list[tuple[int, str, int, str]],  # [(stage_i, run_i_path, stage_j, run_j_path)]
        camera_names: Sequence[str],
        history_len: int,
        skip_frame: int,
        offset: int,
        stage_embeddings: dict[int, list[float]] | None = None,
        stage_texts: dict[int, str] | None = None,
        use_augmentation: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.camera_names = list(camera_names)
        self.history_len = history_len
        self.skip_frame = skip_frame
        self.offset = offset
        self.stage_embeddings = stage_embeddings or {}
        self.stage_texts = stage_texts or {}
        self.use_augmentation = use_augmentation
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size, use_augmentation=True
            )

    def __len__(self) -> int:
        return len(self.pairs)

    def _list_cam_files(self, run_path: str, cam: str) -> list[str]:
        cam_dir = os.path.join(run_path, cam)
        files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_full.jpg")])
        if not files:
            raise RuntimeError(f"No *_full.jpg under {cam_dir}")
        return [os.path.join(cam_dir, f) for f in files]

    def __getitem__(self, index: int):
        stage_i, run_i, stage_j, run_j = self.pairs[index]

        # Build file lists
        cam_to_files_i = {cam: self._list_cam_files(run_i, cam) for cam in self.camera_names}
        cam_to_files_j = {cam: self._list_cam_files(run_j, cam) for cam in self.camera_names}
        len_i = len(next(iter(cam_to_files_i.values())))
        len_j = len(next(iter(cam_to_files_j.values())))
        max_pair_len = len_i + len_j

        # Sample current and target within concatenated pair
        min_start = self.history_len * self.skip_frame
        max_start = max_pair_len - self.offset - 1
        if max_start < min_start:
            # Not enough timeline to sample; resample another index
            # Simple fallback: clamp to earliest feasible configuration
            curr = min_start
        else:
            curr = int(np.random.randint(min_start, max_start + 1))
        target_ts = curr + self.offset

        # Build history indices
        hist_indices = list(range(curr - self.history_len * self.skip_frame, curr + 1, self.skip_frame))

        # Load images for each camera and timestep
        image_sequence_per_cam = []
        for cam in self.camera_names:
            frames = []
            files_i = cam_to_files_i[cam]
            files_j = cam_to_files_j[cam]
            for ts in hist_indices:
                if ts < len_i:
                    path = files_i[min(ts, len_i - 1)]
                else:
                    idx = ts - len_i
                    path = files_j[min(idx, len_j - 1)]
                img = Image.open(path).convert("RGB")
                if self.use_augmentation:
                    img_np = np.array(img)
                    img_np = self.image_preprocessor.augment_image(img_np, training=True)
                    img = Image.fromarray(img_np)
                frames.append(self.to_tensor(img))
            # Stack per camera: (T, C, H, W)
            image_sequence_per_cam.append(torch.stack(frames, dim=0))

        # Stack cameras: (T, K, C, H, W)
        image_sequence = torch.stack(image_sequence_per_cam, dim=1)

        # Determine target stage by target_ts position
        if target_ts < len_i:
            stage_target = stage_i
        else:
            stage_target = stage_j
        emb_list = self.stage_embeddings.get(stage_target, None)
        if emb_list is None:
            command_embedding = torch.zeros(768, dtype=torch.float32)
        else:
            command_embedding = torch.tensor(emb_list, dtype=torch.float32).squeeze()
        command_text = self.stage_texts.get(stage_target, f"stage_{stage_target}")

        return image_sequence, command_embedding, command_text


def load_paired_stage_sequences(
    root_dir: str,
    camera_names: Sequence[str],
    batch_size: int,
    history_len: int,
    skip_frame: int,
    offset: int,
    stage_embeddings_file: str | None,
    stage_texts_file: str | None = None,
    max_run: int | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
    rng_seed: int | None = None,
):
    """
    Build a DataLoader that, per epoch, samples (max_stage-1) * max_run paired sequences:
      - For each adjacent stage pair (i, i+1):
          - Enumerate up to max_run runs from stage i, and match each with a unique random run from stage i+1.
      - Sampling inside each item respects current+offset < max_pair_length.
    """
    stage_to_runs = _list_stage_runs(root_dir)
    max_stage = max(stage_to_runs.keys())
    stage_embeddings = _collect_stage_embeddings(stage_embeddings_file)
    stage_texts = _collect_stage_texts(stage_texts_file)

    # Optional seed to vary pairing across epochs
    if rng_seed is not None:
        np.random.seed(rng_seed)

    pairs: list[tuple[int, str, int, str]] = []
    for i in range(1, max_stage):
        runs_i = stage_to_runs.get(i, [])
        runs_j = stage_to_runs.get(i + 1, [])
        if not runs_i or not runs_j:
            print(f"[paired] skip stage pair ({i},{i+1}) due to missing runs: "
                  f"runs_i={len(runs_i)}, runs_j={len(runs_j)}")
            continue
        # Determine per-pair quota
        quota = len(runs_i)
        if max_run is not None:
            quota = min(quota, max_run)
        # For uniqueness, shuffle stage i+1 runs and map 1-1 as much as possible
        rng_perm = np.random.permutation(len(runs_j))
        # If stage i has more than stage i+1, wrap around
        assigned = 0
        for idx_i in range(quota):
            run_i = runs_i[idx_i % len(runs_i)]
            idx_j = rng_perm[idx_i % len(runs_j)]
            run_j = runs_j[idx_j]
            # Ensure the concatenated pair has at least history_len*skip + offset + 1 frames
            try:
                len_i = _count_frames_in_run(run_i, camera_names)
                len_j = _count_frames_in_run(run_j, camera_names)
            except Exception as e:
                print(f"[paired] skip pair: stage {i} run {os.path.basename(run_i)} "
                      f"+ stage {i+1} run {os.path.basename(run_j)} due to error: {e}")
                continue
            required = history_len * skip_frame + offset + 1
            if len_i + len_j < required:
                print(f"[paired] skip pair (too short): stage {i} run {os.path.basename(run_i)} "
                      f"(len={len_i}) + stage {i+1} run {os.path.basename(run_j)} (len={len_j}) "
                      f"< required {required}")
                continue
            pairs.append((i, run_i, i + 1, run_j))
            assigned += 1
            if assigned >= quota:
                break

    dataset = PairedStageSequenceDataset(
        pairs=pairs,
        camera_names=camera_names,
        history_len=history_len,
        skip_frame=skip_frame,
        offset=offset,
        stage_embeddings=stage_embeddings,
        stage_texts=stage_texts,
        use_augmentation=use_augmentation,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        prefetch_factor=8,
        persistent_workers=True,
    )
    return loader

class SequentialStageSequenceDataset(torch.utils.data.Dataset):
    """
    Concatenate runs across all stages for the same run name in ascending stage order:
      stage1/run_X -> stage2/run_X -> ... -> stageN/run_X
    Sampling uses random current_frame, history (skip_frame), and target at offset along the concatenated timeline.
    """
    def __init__(
        self,
        sequences: list[tuple[list[int], list[str]]],  # [( [stages...], [run_paths aligned...] )]
        camera_names: Sequence[str],
        history_len: int,
        skip_frame: int,
        offset: int,
        stage_embeddings: dict[int, list[float]] | None = None,
        stage_texts: dict[int, str] | None = None,
        use_augmentation: bool = False,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.sequences = sequences
        self.camera_names = list(camera_names)
        self.history_len = history_len
        self.skip_frame = skip_frame
        self.offset = offset
        self.stage_embeddings = stage_embeddings or {}
        self.stage_texts = stage_texts or {}
        self.use_augmentation = use_augmentation
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size, use_augmentation=True
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def _list_cam_files(self, run_path: str, cam: str) -> list[str]:
        cam_dir = os.path.join(run_path, cam)
        files = sorted([f for f in os.listdir(cam_dir) if f.endswith("_full.jpg")])
        if not files:
            raise RuntimeError(f"No *_full.jpg under {cam_dir}")
        return [os.path.join(cam_dir, f) for f in files]

    def __getitem__(self, index: int):
        stages, run_paths = self.sequences[index]
        # Build camera file arrays for each segment
        cam_files_per_stage = []
        lengths = []
        for run_path in run_paths:
            cam_map = {cam: self._list_cam_files(run_path, cam) for cam in self.camera_names}
            cam_files_per_stage.append(cam_map)
            lengths.append(len(next(iter(cam_map.values()))))
        cumulative = np.cumsum([0] + lengths)  # boundaries
        total_len = cumulative[-1]

        # Sample current within total timeline
        min_start = self.history_len * self.skip_frame
        max_start = total_len - self.offset - 1
        if max_start < min_start:
            curr = min_start
        else:
            curr = int(np.random.randint(min_start, max_start + 1))
        target_ts = curr + self.offset
        hist_indices = list(range(curr - self.history_len * self.skip_frame, curr + 1, self.skip_frame))

        # Build frames
        image_sequence_per_cam = []
        for cam in self.camera_names:
            frames = []
            for ts in hist_indices:
                # find which stage segment ts falls into
                seg_idx = int(np.searchsorted(cumulative, ts, side="right") - 1)
                seg_offset = ts - cumulative[seg_idx]
                files = cam_files_per_stage[seg_idx][cam]
                path = files[min(seg_offset, len(files) - 1)]
                img = Image.open(path).convert("RGB")
                if self.use_augmentation:
                    img_np = np.array(img)
                    img_np = self.image_preprocessor.augment_image(img_np, training=True)
                    img = Image.fromarray(img_np)
                frames.append(self.to_tensor(img))
            image_sequence_per_cam.append(torch.stack(frames, dim=0))
        image_sequence = torch.stack(image_sequence_per_cam, dim=1)  # (T,K,C,H,W)

        # Determine target stage by target_ts
        seg_idx_t = int(np.searchsorted(cumulative, target_ts, side="right") - 1)
        stage_target = stages[seg_idx_t]
        emb_list = self.stage_embeddings.get(stage_target, None)
        if emb_list is None:
            command_embedding = torch.zeros(768, dtype=torch.float32)
        else:
            command_embedding = torch.tensor(emb_list, dtype=torch.float32).squeeze()
        command_text = self.stage_texts.get(stage_target, f"stage_{stage_target}")
        return image_sequence, command_embedding, command_text

def load_sequential_stage_sequences(
    root_dir: str,
    camera_names: Sequence[str],
    batch_size: int,
    history_len: int,
    skip_frame: int,
    offset: int,
    stage_embeddings_file: str | None,
    stage_texts_file: str | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
):
    """
    Build DataLoader where each sample is a sequence created by concatenating the same run name
    across all stages in ascending order: stage1/run_X -> stage2/run_X -> ... -> stageN/run_X.
    """
    stage_to_runs = _list_stage_runs(root_dir)
    max_stage = max(stage_to_runs.keys())
    # Build mapping stage -> set of run names
    stage_to_runnames: dict[int, set[str]] = {}
    for s, run_paths in stage_to_runs.items():
        names = set(os.path.basename(p) for p in run_paths)
        stage_to_runnames[s] = names
    # Find run names present in all stages
    common_names = None
    for s in range(1, max_stage + 1):
        if s not in stage_to_runnames:
            continue
        if common_names is None:
            common_names = set(stage_to_runnames[s])
        else:
            common_names &= stage_to_runnames[s]
    if not common_names:
        raise RuntimeError("No common run names found across all stages")

    # Build sequences list
    sequences: list[tuple[list[int], list[str]]] = []
    for run_name in sorted(common_names):
        stages_order = []
        paths_order = []
        ok = True
        for s in range(1, max_stage + 1):
            candidates = [p for p in stage_to_runs.get(s, []) if os.path.basename(p) == run_name]
            if not candidates:
                ok = False
                break
            stages_order.append(s)
            paths_order.append(candidates[0])
        if ok:
            # Quick feasibility check: ensure total len allows some sampling
            try:
                total_len = sum(_count_frames_in_run(p, camera_names) for p in paths_order)
            except Exception:
                continue
            if total_len >= history_len * skip_frame + offset + 1:
                sequences.append((stages_order, paths_order))

    stage_embeddings = _collect_stage_embeddings(stage_embeddings_file)
    stage_texts = _collect_stage_texts(stage_texts_file)
    dataset = SequentialStageSequenceDataset(
        sequences=sequences,
        camera_names=camera_names,
        history_len=history_len,
        skip_frame=skip_frame,
        offset=offset,
        stage_embeddings=stage_embeddings,
        stage_texts=stage_texts,
        use_augmentation=use_augmentation,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # deterministic order for validation
        pin_memory=True,
        num_workers=8,
        prefetch_factor=8,
        persistent_workers=True,
    )
    return loader
