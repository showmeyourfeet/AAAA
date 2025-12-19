import numpy as np
import torch
import os
import h5py
import cv2
import json
import re
import math
from bisect import bisect_left, bisect_right
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from typing import Sequence
import albumentations as A

from llp.utils import random_crop
from llp.dataset import DAggerSampler


class ImagePreprocessor:
    """Image preprocessor with data augmentation support.
    
    Based on the ImagePreprocessor from srt-h/utils.py, adapted for HighLevelModel training.
    """
    def __init__(
        self,
        image_size: int = 224,
        use_augmentation: bool = True,
        normalize: bool = False,
    ):
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.normalize = normalize

        # Define Albumentations enhancement process
        if use_augmentation:
            # Use ReplayCompose to ensure consistent augmentation across multiple cameras
            self.albumentations_transform = A.ReplayCompose([
                # Don't resize here, keep original size for now
                # Use multiple enhancement methods
                # A.OneOf([
                #     A.Rotate(limit=[-10, 10], p=0.5),
                #     A.Affine(rotate=[-10, 10], scale=[0.9, 1.1], translate_percent=[-0.1, 0.1], shear=[-10, 10], p=0.8),
                # ], p=0.5),
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

    def augment_images(
        self,
        images: list[np.ndarray],
        training: bool,
        replay: dict | None = None,
        return_replay: bool = False,
    ) -> list[np.ndarray] | tuple[list[np.ndarray], dict | None]:
        """Apply consistent augmentation to a list of numpy array images.

        Args:
            images: List of images (one per camera for a timestep).
            training: Whether we are in training mode.
            replay: Optional pretrained replay dict. If provided, reuse exact
                augmentation parameters (used for sequence-level synchronization).
            return_replay: If True, also return the replay dict used.
        """
        if (
            not training
            or not self.use_augmentation
            or self.albumentations_transform is None
            or not images
        ):
            if return_replay:
                return images, replay
            return images

        start_index = 0
        if replay is None:
            augmented_0 = self.albumentations_transform(image=images[0])
            replay = augmented_0.get("replay")
            if replay is None:
                results = [augmented_0["image"]]
                for img in images[1:]:
                    results.append(self.albumentations_transform(image=img)["image"])
                if return_replay:
                    return results, None
                return results
            results = [augmented_0["image"]]
            start_index = 1
        else:
            results = []

        for img in images[start_index:]:
            augmented = A.ReplayCompose.replay(replay, image=img)
            results.append(augmented["image"])
        if return_replay:
            return results, replay
        return results

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
        sync_sequence_augmentation: bool = False,
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
        self.sync_sequence_augmentation = sync_sequence_augmentation
        
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
            sequence_replay = None
            for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
                current_ts_images = []
                for cam_name in self.camera_names:
                    image_data = root[f"/observations/images/{cam_name}"][ts]
                    if compressed:
                        decompressed_image = cv2.imdecode(image_data, 1)
                        image_data = np.array(decompressed_image)
                        if self.random_crop:
                            image_data = random_crop(image_data)
                    image_data = cv2.cvtColor(
                        image_data, cv2.COLOR_BGR2RGB
                    )
                    current_ts_images.append(image_data)
                
                # Apply data augmentation if enabled (synchronized across cameras)
                if self.use_augmentation:
                    if self.sync_sequence_augmentation:
                        current_ts_images, sequence_replay = self.image_preprocessor.augment_images(
                            current_ts_images,
                            self.training,
                            replay=sequence_replay,
                            return_replay=True,
                        )
                    else:
                        current_ts_images = self.image_preprocessor.augment_images(
                            current_ts_images, self.training
                        )
                
                all_cam_images = np.stack(current_ts_images, axis=0)
                image_sequence.append(all_cam_images)

            image_sequence = np.array(image_sequence)
            image_sequence = torch.tensor(image_sequence, dtype=torch.float32)
            image_sequence = torch.einsum("t k h w c -> t k c h w", image_sequence)
            image_sequence = image_sequence / 255.0

        return image_sequence, command_embedding, command_gt


class AnnotationStageDataset(torch.utils.data.Dataset):
    """
    Dataset that samples frame histories from run directories that contain:
      - `frames/` (all jpg files named by millisecond timestamps)
      - `annotations.json` describing stage segments with `start_ms` / `end_ms`.

    Sampling strategy matches `SequenceDataset`: pick a valid current frame,
    build history via `history_len` / `history_skip_frame`, and label the target
    frame (current + `prediction_offset`) by locating the stage that contains
    the target timestamp. Stage metadata (embeddings + texts) can be supplied
    through `stage_embeddings_file` / `stage_texts_file`. Use `drop_initial_frames`
    to discard the first N frames of every run when the capture has unusable
    leading data.
    """

    def __init__(
        self,
        dataset_dir: str,
        run_ids: Sequence[int | str] | None = None,
        history_len: int = 5,
        prediction_offset: int = 10,
        history_skip_frame: int = 1,
        traverse_full_trajectory: bool = False,
        use_augmentation: bool = False,
        training: bool = True,
        image_size: int = 224,
        sync_sequence_augmentation: bool = False,
        stage_embeddings_file: str | None = None,
        stage_texts_file: str | None = None,
        drop_initial_frames: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.history_len = history_len
        self.prediction_offset = prediction_offset
        self.history_skip_frame = history_skip_frame
        self.traverse_full_trajectory = traverse_full_trajectory
        self.use_augmentation = use_augmentation
        self.training = training
        self.image_size = image_size
        self.sync_sequence_augmentation = sync_sequence_augmentation
        self.drop_initial_frames = max(drop_initial_frames, 0)
        self.required_history = self.history_len * self.history_skip_frame

        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation,
            )

        self.stage_embeddings, self.stage_texts = self._load_stage_metadata(
            stage_embeddings_file, stage_texts_file
        )
        self.embedding_dim = (
            len(next(iter(self.stage_embeddings.values())))
            if self.stage_embeddings
            else 0
        )

        self.runs = self._build_run_records(run_ids)
        # Pre-compute full-trajectory indices if needed
        self._full_traj_index: list[tuple[int, int]] = []
        if self.traverse_full_trajectory:
            self._full_traj_index = self._build_full_trajectory_index()

        if not self.runs:
            raise ValueError(
                "No valid runs found. "
                "Check that frames and annotations exist and align correctly."
            )

    def __len__(self) -> int:
        if self.traverse_full_trajectory:
            # One sample per valid timestep across all runs
            return len(self._full_traj_index)
        # Match SequenceDataset semantics: one sample per run per epoch
        return len(self.runs)

    def __getitem__(self, index: int):
        if self.traverse_full_trajectory:
            if not self._full_traj_index:
                raise ValueError(
                    "No valid indices for full-trajectory traversal. "
                    "Check that runs have enough frames for the given history_len and prediction_offset."
                )
            sample_idx = index % len(self._full_traj_index)
            run_idx, curr_idx = self._full_traj_index[sample_idx]
            run_rec = self.runs[run_idx]
            target_idx = curr_idx + self.prediction_offset
            hist_start = curr_idx - self.required_history
        else:
            run_idx = index % len(self.runs)
            run_rec = self.runs[run_idx]

            total_frames = len(run_rec["frames"])
            try:
                curr_idx = np.random.randint(
                    self.required_history,
                    total_frames - self.prediction_offset,
                )
                target_idx = curr_idx + self.prediction_offset
                hist_start = curr_idx - self.required_history
            except ValueError:
                # 当前 run 太短，换一个 run 重试
                return self.__getitem__((index + 1) % len(self))
        history_indices = range(
            hist_start, curr_idx + 1, self.history_skip_frame
        )

        image_sequence = []
        sequence_replay = None
        for frame_idx in history_indices:
            _timestamp, frame_path = run_rec["frames"][frame_idx]
            current_ts_images = []
            img_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to load frame: {frame_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            current_ts_images.append(img_rgb)

            if self.use_augmentation:
                if self.sync_sequence_augmentation:
                    current_ts_images, sequence_replay = self.image_preprocessor.augment_images(
                        current_ts_images,
                        self.training,
                        replay=sequence_replay,
                        return_replay=True,
                    )
                else:
                    current_ts_images = self.image_preprocessor.augment_images(
                        current_ts_images, self.training
                    )

            all_cam_images = np.stack(current_ts_images, axis=0)
            image_sequence.append(all_cam_images)

        image_sequence = np.array(image_sequence)
        image_sequence = torch.tensor(image_sequence, dtype=torch.float32)
        image_sequence = torch.einsum("t k h w c -> t k c h w", image_sequence)
        image_sequence = image_sequence / 255.0

        target_stage = self._stage_for_index(run_rec, target_idx)
        if target_stage is None:
            # 若 target 帧没有落在任何标注段内，则尝试换下一个 run
            try:
                return self.__getitem__((index + 1) % len(self))
            except RecursionError:
                print(
                    f"RecursionError: Could not find stage for run {os.path.basename(run_rec['run_dir'])} "
                    f"and target_idx {target_idx}."
                )
                raise

        stage_id = target_stage["stage_id"]
        stage_name = target_stage["stage_name"]
        command_embedding = self._embedding_for_stage(stage_id)

        return image_sequence, command_embedding, stage_name

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _normalize_run_name(self, run_id: int | str) -> str:
        if isinstance(run_id, int):
            return f"run_{run_id:03d}"
        run_str = str(run_id)
        if not run_str.startswith("run_"):
            return f"run_{run_str}"
        return run_str

    def _discover_runs(self) -> list[str]:
        runs = []
        for entry in sorted(os.listdir(self.dataset_dir)):
            candidate = os.path.join(self.dataset_dir, entry)
            if os.path.isdir(candidate) and entry.startswith("run_"):
                runs.append(candidate)
        return runs

    def _build_run_records(self, run_ids: Sequence[int | str] | None):
        if run_ids:
            run_dirs = []
            for run_id in run_ids:
                name = self._normalize_run_name(run_id)
                path = os.path.join(self.dataset_dir, name)
                if os.path.isdir(path):
                    run_dirs.append(path)
            if not run_dirs:
                raise ValueError(
                    f"No matching runs found under {self.dataset_dir} "
                    f"for ids: {run_ids}"
                )
        else:
            run_dirs = self._discover_runs()
        records = []
        for path in run_dirs:
            frames = self._load_frames(path)
            if not frames:
                continue
            timestamps = [ts for ts, _ in frames]
            stages = self._load_stage_segments(path, timestamps)
            if not stages:
                continue
            records.append(
                {
                    "run_dir": path,
                    "frames": frames,
                    "timestamps": timestamps,
                    "stages": stages,
                }
            )
        return records

    def _build_sampling_space(self):
        sampling = []
        for idx, run_rec in enumerate(self.runs):
            total_frames = len(run_rec["frames"])
            max_curr_global = total_frames - self.prediction_offset - 1
            if max_curr_global <= self.required_history:
                continue
            for stage in run_rec["stages"]:
                min_curr = max(stage["start_idx"], self.required_history)
                max_curr = min(stage["end_idx"] - 1, max_curr_global)
                if min_curr > max_curr:
                    continue
                sampling.append(
                    {
                        "run_index": idx,
                        "stage": stage,
                        "min_curr": min_curr,
                        "max_curr": max_curr,
                    }
                )
        return sampling

    def _build_full_trajectory_index(self) -> list[tuple[int, int]]:
        """
        Build a flattened list of (run_index, curr_idx) pairs that
        traverse each trajectory from:
          curr_idx in [required_history + 1, total_frames - prediction_offset - 1]
        This matches the user's requirement:
          start_ts 从 history_len * history_skip_frame + 1
          一直到 total_len - prediction_offset - 1，遍历整条轨迹.
        """
        index: list[tuple[int, int]] = []
        for run_idx, run_rec in enumerate(self.runs):
            total_frames = len(run_rec["frames"])
            # 下界: 至少有完整的历史（包含 curr_idx == required_history）
            start_curr = self.required_history
            # 上界: 目标帧 + offset 不能越界
            end_curr = total_frames - self.prediction_offset - 1
            if end_curr < start_curr:
                continue
            for curr_idx in range(start_curr, end_curr + 1):
                index.append((run_idx, curr_idx))
        return index

    def _load_frames(self, run_dir: str):
        frames_dir = os.path.join(run_dir, "frames")
        if not os.path.isdir(frames_dir):
            return []
        frames = []
        for file_name in os.listdir(frames_dir):
            if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(file_name)[0]
            try:
                timestamp = int(stem)
            except ValueError:
                continue
            frames.append((timestamp, os.path.join(frames_dir, file_name)))
        frames.sort(key=lambda item: item[0])
        if self.drop_initial_frames > 0 and len(frames) > self.drop_initial_frames:
            frames = frames[self.drop_initial_frames :]
        return frames

    def _load_stage_segments(self, run_dir: str, timestamps: list[int]):
        ann_path = os.path.join(run_dir, "annotations.json")
        if not os.path.exists(ann_path):
            return []
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)
        stage_lookup = {
            int(stage["id"]): stage.get("name", f"stage_{stage['id']}")
            for stage in ann_data.get("stages", [])
            if "id" in stage
        }
        segments = []
        for entry in ann_data.get("annotations", []):
            if "start_ms" not in entry or "end_ms" not in entry:
                continue
            stage_id = int(entry.get("stage_id", -1))
            start_ms = int(entry["start_ms"])
            end_ms = int(entry["end_ms"])
            start_idx = bisect_left(timestamps, start_ms)
            end_idx = bisect_right(timestamps, end_ms)
            if end_idx - start_idx <= 0:
                continue
            segments.append(
                {
                    "stage_id": stage_id,
                    "stage_name": entry.get(
                        "stage_name",
                        stage_lookup.get(
                            stage_id,
                            self.stage_texts.get(
                                stage_id, f"stage_{stage_id}"
                            ),
                        ),
                    ),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                }
            )
        return sorted(segments, key=lambda seg: seg["start_idx"])

    def _load_stage_metadata(
        self,
        embeddings_file: str | None,
        texts_file: str | None,
    ):
        stage_embeddings = _collect_stage_embeddings(embeddings_file)
        # Prefer explicit texts file; fallback to embeddings file if it also stores text
        text_source = texts_file or embeddings_file
        stage_texts = _collect_stage_texts(text_source)
        return stage_embeddings, stage_texts

    def _stage_for_index(self, run_rec, frame_idx: int):
        for stage in run_rec["stages"]:
            if stage["start_idx"] <= frame_idx < stage["end_idx"]:
                return stage
        return None

    def _embedding_for_stage(self, stage_id: int):
        emb = self.stage_embeddings.get(stage_id)
        if emb is None:
            if self.embedding_dim == 0:
                return torch.zeros(0, dtype=torch.float32)
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        return torch.tensor(emb, dtype=torch.float32).squeeze()


def load_annotation_data(
    root_dir: str,
    batch_size_train: int,
    batch_size_val: int,
    history_len: int = 5,
    prediction_offset: int = 10,
    history_skip_frame: int = 1,
    traverse_full_trajectory: bool = False,
    num_workers_train: int = 0,
    num_workers_val: int = 0,
    stage_embeddings_file: str | None = None,
    stage_texts_file: str | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
    drop_initial_frames: int = 1,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    seed: int = 0,
):
    """
    Build train/val/test DataLoaders for annotation-based runs:
    root_dir/run_xxx/{frames/, annotations.json}.
    """
    # Discover available runs
    all_runs = [
        d
        for d in sorted(os.listdir(root_dir))
        if d.startswith("run_") and os.path.isdir(os.path.join(root_dir, d))
    ]
    if not all_runs:
        raise ValueError(f"No run_* directories found under {root_dir}")

    # Deterministic split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_runs))
    all_runs = [all_runs[i] for i in perm]

    n_total = len(all_runs)

    # Use rounding for splits, with safeguards to ensure non-empty train
    # and optional non-empty val if val_ratio > 0.
    if val_ratio > 0 and n_total >= 2:
        n_val = int(round(val_ratio * n_total))
        if n_val == 0:
            n_val = 1
    else:
        n_val = 0

    remaining = max(0, n_total - n_val)
    if train_ratio > 0 and remaining >= 1:
        n_train = int(round(train_ratio * n_total))
        if n_train == 0:
            n_train = 1
        if n_train > remaining:
            n_train = remaining
    else:
        n_train = remaining

    # Ensure total does not exceed n_total
    if n_train + n_val > n_total:
        # Prefer to keep at least 1 train sample
        overflow = n_train + n_val - n_total
        reduce_val = min(overflow, max(0, n_val - 1))
        n_val -= reduce_val
        overflow -= reduce_val
        if overflow > 0:
            n_train = max(1, n_train - overflow)

    n_test = max(0, n_total - n_train - n_val)
    train_runs = all_runs[:n_train]
    val_runs = all_runs[n_train : n_train + n_val]
    test_runs = all_runs[n_train + n_val :] if n_test > 0 else []

    def _build_dataset(run_ids, training: bool, aug: bool):
        return AnnotationStageDataset(
            dataset_dir=root_dir,
            run_ids=run_ids if run_ids else None,
            history_len=history_len,
            prediction_offset=prediction_offset,
            history_skip_frame=history_skip_frame,
            traverse_full_trajectory=traverse_full_trajectory,
            use_augmentation=aug,
            training=training,
            image_size=image_size,
            sync_sequence_augmentation=False,
            stage_embeddings_file=stage_embeddings_file,
            stage_texts_file=stage_texts_file,
            drop_initial_frames=drop_initial_frames,
        )

    train_dataset = _build_dataset(train_runs, training=True, aug=use_augmentation)
    val_dataset = _build_dataset(val_runs, training=False, aug=False) if val_runs else None
    # Build test dataset only if we have dedicated test runs
    test_dataset = (
        _build_dataset(test_runs, training=False, aug=False) if test_runs else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers_train,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size_val,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers_val,
        )
        if val_dataset is not None
        else None
    )
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


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
    sync_sequence_augmentation=False,
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
            sync_sequence_augmentation=sync_sequence_augmentation,
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
            sync_sequence_augmentation=sync_sequence_augmentation,
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
            sync_sequence_augmentation=sync_sequence_augmentation,
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
        sync_sequence_augmentation: bool = False,
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
        self.sync_sequence_augmentation = sync_sequence_augmentation
        
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
        
        # Note: target_stage_idx is 0-based internally, but stage_embeddings/stage_texts use 1-based keys
        stage_key = target_stage_idx + 1
        command_embedding = self.stage_embeddings.get(stage_key, None)
        if command_embedding is None:
            command_embedding = torch.zeros(768)
        else:
            command_embedding = torch.tensor(command_embedding).float().squeeze()
        
        command_gt = self.stage_texts.get(stage_key, f"stage_{target_stage_idx}")
        
        # Load image sequence for history frames
        image_sequence = []
        for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
            current_ts_images = []
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
                if self.random_crop:
                    img_np = random_crop(img_np)
                
                current_ts_images.append(img_np)
            
            # Apply data augmentation if enabled (synchronized across cameras)
            if self.use_augmentation:
                current_ts_images = self.image_preprocessor.augment_images(
                    current_ts_images, self.training
                )
            
            # Convert to tensor format (C, H, W) in [0, 1]
            current_ts_tensors = [self.to_tensor(img) for img in current_ts_images]
            all_cam_images = torch.stack(current_ts_tensors, dim=0)
            image_sequence.append(all_cam_images)
        
        # Stack sequence: (timesteps, num_cameras, C, H, W)
        image_sequence = torch.stack(image_sequence, dim=0)
        
        return image_sequence, command_embedding, command_gt


class CompositeSequenceDataset(torch.utils.data.Dataset):
    """
    Composite dataset that combines the advantages of SequenceDataset and SplittedSequenceDataset.
    
    For each sample:
    1. Selects one run from each stage (either from pre-generated combinations or randomly)
    2. Concatenates these runs into a complete sequence covering all stages
    3. Samples from this concatenated sequence using the standard sampling pattern
       (random current_frame -> start_frame = current_frame - skip_frame*history_len,
        target_frame = current_frame + offset)
    
    This provides:
    - Complete sequences covering all stages (like SequenceDataset)
    - Balanced coverage of all runs (via pre-generated combinations)
    - Optional randomness for data augmentation
    
    Sampling strategies:
    - 'balanced': Pre-generate combinations ensuring each run is used roughly equally
    - 'random': Fully random selection each time (may have imbalanced coverage)
    - 'sequential': Sequential matching by run number (run_001+run_001+..., run_002+run_002+...)
                   Ideal for validation/test sets for deterministic evaluation
    """
    def __init__(
        self,
        root_dir: str,
        camera_names,
        history_len=4,
        prediction_offset=15,
        history_skip_frame=20,
        random_crop=False,
        stage_embeddings: dict | None = None,
        stage_texts: dict | None = None,
        use_augmentation=False,
        training=True,
        image_size=224,
        sampling_strategy='balanced',  # 'balanced' or 'random' or 'sequential'
        max_combinations=None,  # Maximum number of combinations (if None, use n_repeats)
        n_repeats=None,  # Number of times each run should be used (alternative to max_combinations)
        sync_sequence_augmentation: bool = False,
        use_weak_traversal: bool = False,  # Use weak traversal sampling strategy
        samples_non_cross_per_stage: int = 1,  # Number of non-cross-stage samples per stage (target_ts in current stage)
        samples_cross_per_stage: int = 1,  # Number of cross-stage samples per stage (target_ts in next stage)
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
        self.sampling_strategy = sampling_strategy
        self.sync_sequence_augmentation = sync_sequence_augmentation
        self.use_weak_traversal = use_weak_traversal
        self.samples_non_cross_per_stage = samples_non_cross_per_stage
        self.samples_cross_per_stage = samples_cross_per_stage
        
        # Initialize ImagePreprocessor if augmentation is enabled
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation
            )
        
        # Organize runs by stage: {stage_idx: [list of run_paths]}
        self.runs_by_stage = {}
        for stage_dir in sorted(os.listdir(root_dir)):
            stage_path = os.path.join(root_dir, stage_dir)
            if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                continue
            try:
                stage_idx = int(stage_dir[5:]) - 1  # stage1 -> 0, stage2 -> 1, etc.
            except Exception:
                continue
            
            runs = []
            for run_dir in sorted(os.listdir(stage_path)):
                run_path = os.path.join(stage_path, run_dir)
                if run_dir.startswith("run") and os.path.isdir(run_path):
                    runs.append(run_path)
            
            if len(runs) > 0:
                self.runs_by_stage[stage_idx] = runs
        
        # Get sorted stage indices
        self.stage_indices = sorted(self.runs_by_stage.keys())
        
        if len(self.stage_indices) == 0:
            print(f"[CompositeSequenceDataset] No runs found under {root_dir}")
        
        # Calculate number of possible combinations (product of runs per stage)
        self.num_combinations = 1
        for stage_idx in self.stage_indices:
            self.num_combinations *= len(self.runs_by_stage[stage_idx])
        
        # Determine max_combinations from n_repeats if provided
        if max_combinations is None and n_repeats is not None:
            # Calculate based on n_repeats: max_combinations = n × min_runs_per_stage
            min_runs = min(len(self.runs_by_stage[s]) for s in self.stage_indices) if self.stage_indices else 1
            max_combinations = n_repeats * min_runs
            print(f"[CompositeSequenceDataset] Calculated max_combinations = {n_repeats} × {min_runs} = {max_combinations}")
        elif max_combinations is None:
            # Default to 50000 if neither is specified
            max_combinations = 50000
        
        # Generate combinations based on sampling strategy
        if sampling_strategy == 'balanced':
            # Pre-generate balanced combinations ensuring each run is used roughly equally
            self.run_combinations = self._generate_balanced_combinations(max_combinations)
            num_combinations = len(self.run_combinations)
        elif sampling_strategy == 'sequential':
            # Pre-generate sequential combinations matching runs by their number
            self.run_combinations = self._generate_sequential_combinations()
            num_combinations = len(self.run_combinations)
        else:  # 'random'
            # Use random sampling on-the-fly
            self.run_combinations = None
            num_combinations = min(max_combinations, self.num_combinations)
        
        # Pre-compute sampling positions for weak traversal strategy
        if use_weak_traversal:
            self.sampling_positions = self._precompute_weak_traversal_sampling(num_combinations)
            self.dataset_size = len(self.sampling_positions)
            print(f"[CompositeSequenceDataset] Using weak traversal strategy")
            print(f"[CompositeSequenceDataset] Generated {num_combinations} combinations")
            print(f"[CompositeSequenceDataset] With {samples_non_cross_per_stage} non-cross + {samples_cross_per_stage} cross samples per stage, total dataset size: {self.dataset_size}")
        else:
            self.sampling_positions = None
            # For non-weak-traversal, use simple random sampling (1 sample per combination)
            self.dataset_size = num_combinations
            print(f"[CompositeSequenceDataset] Generated {num_combinations} combinations")
            print(f"[CompositeSequenceDataset] Total dataset size: {self.dataset_size}")
        
        self.to_tensor = transforms.ToTensor()
    
    def _generate_balanced_combinations(self, max_combinations):
        """
        Generate combinations ensuring each run is used roughly equally.
        
        Strategy:
        1. Calculate how many times each run should appear to reach max_combinations
        2. Create a balanced list by cycling through runs in each stage
        3. Shuffle the combinations for randomness
        """
        # Calculate target usage per run
        min_runs_per_stage = min(len(self.runs_by_stage[s]) for s in self.stage_indices)
        
        # Determine how many rounds of each run we should include
        # Each round means one full cycle through all runs in a stage
        if self.num_combinations <= max_combinations:
            # Use all possible combinations
            target_samples = self.num_combinations
        else:
            # Limit to max_combinations
            target_samples = max_combinations
        
        combinations = []
        
        # Strategy: Ensure each run appears roughly the same number of times
        # Calculate repetitions needed per stage
        runs_per_stage = [len(self.runs_by_stage[s]) for s in self.stage_indices]
        
        # For balanced coverage, we want each run to appear ~equally
        # Calculate LCM-like approach or use round-robin
        
        # Simple approach: Round-robin cycling ensuring balance
        if len(self.stage_indices) == 2:
            # For 2 stages, use a cycling pattern that ensures balance
            stage0_runs = self.runs_by_stage[self.stage_indices[0]]
            stage1_runs = self.runs_by_stage[self.stage_indices[1]]
            
            # Check if we want all combinations or a subset
            total_possible = len(stage0_runs) * len(stage1_runs)
            
            if target_samples >= total_possible:
                # Generate all combinations (cartesian product)
                for run0 in stage0_runs:
                    for run1 in stage1_runs:
                        combinations.append([
                            (self.stage_indices[0], run0),
                            (self.stage_indices[1], run1)
                        ])
            else:
                # Generate balanced subset using interleaved cycling
                # Strategy: ensure each run appears roughly equally
                # For target_samples = n * R, each run should appear ~n times
                
                R0 = len(stage0_runs)
                R1 = len(stage1_runs)
                
                # Build balanced lists for each stage
                stage0_sequence = []
                stage1_sequence = []
                
                # Calculate how many complete cycles we need
                cycles_needed = (target_samples // min(R0, R1)) + 1
                
                # Create stage0 sequence: repeat each run equally
                for _ in range(cycles_needed):
                    stage0_sequence.extend(stage0_runs)
                    
                # Create stage1 sequence with cycling offset to avoid repetition
                for cycle in range(cycles_needed):
                    # Rotate the list for each cycle to create variety
                    offset = (cycle * 37) % R1  # Prime number for good distribution
                    for i in range(R1):
                        stage1_sequence.append(stage1_runs[(i + offset) % R1])
                
                # Trim to exact target size
                stage0_sequence = stage0_sequence[:target_samples]
                stage1_sequence = stage1_sequence[:target_samples]
                
                # Combine into combinations
                for run0, run1 in zip(stage0_sequence, stage1_sequence):
                    combinations.append([
                        (self.stage_indices[0], run0),
                        (self.stage_indices[1], run1)
                    ])
        else:
            # For arbitrary number of stages, use iterative approach
            import itertools
            
            # Get all runs for each stage
            stage_runs = [self.runs_by_stage[s] for s in self.stage_indices]
            
            # Generate combinations up to max
            if self.num_combinations <= max_combinations:
                # Generate all combinations
                for combo in itertools.product(*stage_runs):
                    combinations.append([
                        (self.stage_indices[i], run) 
                        for i, run in enumerate(combo)
                    ])
            else:
                # Sample combinations with replacement, trying to balance
                rng = np.random.RandomState(42)  # Fixed seed for reproducibility
                for _ in range(target_samples):
                    combo = []
                    for stage_idx in self.stage_indices:
                        run = rng.choice(self.runs_by_stage[stage_idx])
                        combo.append((stage_idx, run))
                    combinations.append(combo)
        
        # Shuffle for randomness across epochs
        rng = np.random.RandomState(42)
        rng.shuffle(combinations)
        
        # Report statistics
        self._report_combination_statistics(combinations)
        
        return combinations
    
    def _generate_sequential_combinations(self):
        """
        Generate sequential combinations matching runs by their number.
        
        Strategy:
        - Extract run numbers from run directory names (e.g., "run_042" -> 42)
        - Match runs with the same number across stages
        - Example: stage1/run_001 + stage2/run_001, stage1/run_002 + stage2/run_002, ...
        
        This ensures deterministic validation where each combination uses 
        the same run number across all stages.
        """
        import re
        
        # Extract run numbers for each stage
        runs_by_number = {}  # {run_number: {stage_idx: run_path}}
        
        for stage_idx in self.stage_indices:
            for run_path in self.runs_by_stage[stage_idx]:
                # Extract run number from path like ".../stage1/run_042"
                match = re.search(r'run[_-]?(\d+)', os.path.basename(run_path))
                if match:
                    run_num = int(match.group(1))
                    if run_num not in runs_by_number:
                        runs_by_number[run_num] = {}
                    runs_by_number[run_num][stage_idx] = run_path
        
        # Create combinations for run numbers that exist in ALL stages
        combinations = []
        for run_num in sorted(runs_by_number.keys()):
            # Check if this run number exists in all stages
            if len(runs_by_number[run_num]) == len(self.stage_indices):
                combo = []
                for stage_idx in self.stage_indices:
                    run_path = runs_by_number[run_num][stage_idx]
                    combo.append((stage_idx, run_path))
                combinations.append(combo)
        
        print(f"[CompositeSequenceDataset] Sequential matching:")
        print(f"  - Total unique run numbers: {len(runs_by_number)}")
        print(f"  - Run numbers present in all stages: {len(combinations)}")
        
        if len(combinations) == 0:
            print(f"  WARNING: No matching run numbers across all stages!")
            print(f"  Falling back to using first available runs from each stage")
            # Fallback: create combinations from first N runs
            min_runs = min(len(self.runs_by_stage[s]) for s in self.stage_indices)
            for i in range(min_runs):
                combo = []
                for stage_idx in self.stage_indices:
                    combo.append((stage_idx, self.runs_by_stage[stage_idx][i]))
                combinations.append(combo)
        
        return combinations
    
    def _report_combination_statistics(self, combinations):
        """Report statistics about run usage in combinations."""
        from collections import Counter
        
        usage_per_stage = {stage_idx: Counter() for stage_idx in self.stage_indices}
        
        for combo in combinations:
            for stage_idx, run_path in combo:
                usage_per_stage[stage_idx][run_path] += 1
        
        print(f"[CompositeSequenceDataset] Combination statistics:")
        for stage_idx in self.stage_indices:
            counts = list(usage_per_stage[stage_idx].values())
            if counts:
                print(f"  Stage {stage_idx}: {len(counts)} unique runs, "
                      f"usage range [{min(counts)}, {max(counts)}], "
                      f"mean {np.mean(counts):.1f}, std {np.std(counts):.1f}")
            else:
                print(f"  Stage {stage_idx}: No runs used")
    
    def _precompute_weak_traversal_sampling(self, num_combinations):
        """
        Pre-compute sampling positions for weak traversal strategy.
        
        For each combination:
        - Traverse stages in order
        - For each stage:
          - Sample samples_non_cross_per_stage times with target_ts in current stage (non-cross-stage)
          - Sample samples_cross_per_stage times with target_ts in next stage (cross-stage)
          - If last stage (no next stage), only non-cross-stage samples are generated
        
        Returns:
            List of (combination_idx, stage_idx_in_combo, is_cross_stage, stage_boundaries_info) tuples
            where stage_boundaries_info is (stage_start, stage_end, next_stage_start, next_stage_end, total_timesteps)
        """
        sampling_positions = []
        
        # Determine which combinations to use
        if self.run_combinations is not None:
            combinations_to_process = self.run_combinations
        else:
            # For random strategy, we'll generate combinations on-the-fly
            # Use a fixed seed to ensure reproducibility
            rng = np.random.RandomState(42)
            combinations_to_process = []
            for combo_idx in range(num_combinations):
                combo = []
                for stage_idx in self.stage_indices:
                    runs = self.runs_by_stage[stage_idx]
                    if len(runs) > 0:
                        selected_run = runs[rng.randint(0, len(runs))]
                        combo.append((stage_idx, selected_run))
                combinations_to_process.append(combo)
        
        min_start = self.history_len * self.history_skip_frame
        
        for combo_idx, selected_runs in enumerate(combinations_to_process):
            # Compute stage boundaries for this combination
            stage_boundaries = []
            run_lengths = []
            cumulative_length = 0
            
            for stage_idx, selected_run in selected_runs:
                labels, actions_np, qpos_np = self._parse_run(selected_run)
                if actions_np is None or len(actions_np) < 1:
                    # Skip this combination if we can't parse it
                    break
                
                run_length = len(actions_np)
                run_lengths.append(run_length)
                stage_boundaries.append((cumulative_length, cumulative_length + run_length))
                cumulative_length += run_length
            
            if len(stage_boundaries) == 0:
                continue
            
            total_timesteps = cumulative_length
            max_start = total_timesteps - self.prediction_offset - 1
            
            if max_start < min_start:
                # Not enough timesteps, skip this combination
                continue
            
            # Traverse each stage
            for stage_idx_in_combo, stage_idx in enumerate(self.stage_indices):
                stage_start, stage_end = stage_boundaries[stage_idx_in_combo]
                
                # Use separate parameters for non-cross and cross-stage samples
                num_non_cross = self.samples_non_cross_per_stage
                
                # Check if cross-stage sampling is possible (not the last stage)
                can_cross_stage = stage_idx_in_combo < len(self.stage_indices) - 1
                
                # Adjust counts if cross-stage is not possible
                if not can_cross_stage:
                    num_cross = 0
                else:
                    num_cross = self.samples_cross_per_stage
                
                # Prepare stage boundaries info
                next_stage_start = stage_boundaries[stage_idx_in_combo + 1][0] if can_cross_stage else stage_end
                next_stage_end = stage_boundaries[stage_idx_in_combo + 1][1] if can_cross_stage else stage_end
                stage_boundaries_info = (stage_start, stage_end, next_stage_start, next_stage_end, total_timesteps)
                
                # Sample non-cross-stage: target_ts should be in current stage
                non_cross_max_curr = min(stage_end - self.prediction_offset - 1, max_start)
                non_cross_min_curr = max(min_start, stage_start)
                
                if non_cross_max_curr >= non_cross_min_curr:
                    # Generate num_non_cross non-cross-stage samples
                    for _ in range(num_non_cross):
                        sampling_positions.append((combo_idx, stage_idx_in_combo, False, stage_boundaries_info))
                
                # Sample cross-stage: target_ts should be in next stage (if exists)
                if can_cross_stage:
                    next_stage_start, next_stage_end = stage_boundaries[stage_idx_in_combo + 1]
                    # curr_ts should be such that: curr_ts + offset >= next_stage_start and < next_stage_end
                    cross_min_curr = max(min_start, next_stage_start - self.prediction_offset)
                    cross_max_curr = min(next_stage_end - self.prediction_offset - 1, max_start)
                    
                    if cross_max_curr >= cross_min_curr:
                        # Generate num_cross cross-stage samples
                        for _ in range(num_cross):
                            sampling_positions.append((combo_idx, stage_idx_in_combo, True, stage_boundaries_info))
        
        return sampling_positions
    
    def __len__(self):
        return self.dataset_size
    
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
        # Use weak traversal if enabled
        if self.use_weak_traversal and self.sampling_positions is not None:
            if index >= len(self.sampling_positions):
                # Fallback to next valid index
                return self.__getitem__((index + 1) % len(self))
            
            combo_idx, stage_idx_in_combo, is_cross_stage, stage_boundaries_info = self.sampling_positions[index]
            stage_start, stage_end, next_stage_start, next_stage_end, total_timesteps = stage_boundaries_info
            
            # Get the combination
            if self.run_combinations is not None:
                selected_runs = self.run_combinations[combo_idx]
            else:
                # For random strategy, generate combination on-the-fly with fixed seed
                rng = np.random.RandomState(combo_idx)
                selected_runs = []
                for stage_idx in self.stage_indices:
                    runs = self.runs_by_stage[stage_idx]
                    if len(runs) == 0:
                        return self.__getitem__((index + 1) % len(self))
                    selected_run = runs[rng.randint(0, len(runs))]
                    selected_runs.append((stage_idx, selected_run))
        else:
            # Original random sampling logic
            combo_idx = index
            
            # Select runs based on sampling strategy
            if self.sampling_strategy == 'balanced' and self.run_combinations is not None:
                # Use pre-generated combination
                selected_runs = self.run_combinations[combo_idx]
            else:
                # Random sampling on-the-fly
                rng = np.random.RandomState(combo_idx)
                selected_runs = []
                for stage_idx in self.stage_indices:
                    runs = self.runs_by_stage[stage_idx]
                    if len(runs) == 0:
                        # Fallback: try another sample
                        return self.__getitem__((index + 1) % len(self))
                    
                    # Randomly select a run from this stage using seeded RNG
                    selected_run = runs[rng.randint(0, len(runs))]
                    selected_runs.append((stage_idx, selected_run))
            
            # Sample a random curr_ts (will be computed after stage_boundaries)
            curr_ts = None
        
        # Now process the selected runs
        stage_boundaries = []  # Track where each stage starts in the concatenated sequence
        run_lengths = []  # Store length of each selected run
        cumulative_length = 0
        
        for stage_idx, selected_run in selected_runs:
            if selected_run is None:
                # Fallback for any issues
                return self.__getitem__((index + 1) % len(self))
            
            # Parse run to get its length
            labels, actions_np, qpos_np = self._parse_run(selected_run)
            if actions_np is None or len(actions_np) < 1:
                # Fallback: try another sample
                return self.__getitem__((index + 1) % len(self))
            
            run_length = len(actions_np)
            run_lengths.append(run_length)
            stage_boundaries.append((cumulative_length, cumulative_length + run_length))
            cumulative_length += run_length
        
        total_timesteps = cumulative_length
        
        # Determine curr_ts based on weak traversal constraints or random sampling
        if self.use_weak_traversal and self.sampling_positions is not None:
            # Use constraints from pre-computed sampling positions
            min_start = self.history_len * self.history_skip_frame
            max_start = total_timesteps - self.prediction_offset - 1
            
            if is_cross_stage:
                # Cross-stage: target_ts should be in next stage
                # curr_ts should be such that: curr_ts + offset >= next_stage_start and < next_stage_end
                cross_min_curr = max(min_start, next_stage_start - self.prediction_offset)
                cross_max_curr = min(next_stage_end - self.prediction_offset - 1, max_start)
                
                if cross_max_curr >= cross_min_curr:
                    # Randomly sample curr_ts within valid range
                    # Uses global random state: each call produces different random value
                    # - Different comps in same epoch: different curr_ts
                    # - Different stages in same comp: different curr_ts
                    # - Multiple samples in same stage (e.g., samples_cross_per_stage=2): 
                    #   Each call to __getitem__ uses np.random.randint(), so each sample gets 
                    #   a different random curr_ts, ensuring independence between samples
                    # - Same index in different epochs: different curr_ts (due to DataLoader shuffle)
                    curr_ts = np.random.randint(cross_min_curr, cross_max_curr + 1)
                else:
                    # Fallback: try another sample
                    return self.__getitem__((index + 1) % len(self))
            else:
                # Non-cross-stage: target_ts should be in current stage
                # curr_ts should be such that: curr_ts + offset < stage_end
                non_cross_max_curr = min(stage_end - self.prediction_offset - 1, max_start)
                non_cross_min_curr = max(min_start, stage_start)
                
                if non_cross_max_curr >= non_cross_min_curr:
                    # Randomly sample curr_ts within valid range
                    # Uses global random state: each call produces different random value
                    # - Different comps in same epoch: different curr_ts
                    # - Different stages in same comp: different curr_ts
                    # - Multiple samples in same stage (e.g., samples_non_cross_per_stage=2): 
                    #   Each call to __getitem__ uses np.random.randint(), so each sample gets 
                    #   a different random curr_ts, ensuring independence between samples
                    # - Same index in different epochs: different curr_ts (due to DataLoader shuffle)
                    curr_ts = np.random.randint(non_cross_min_curr, non_cross_max_curr + 1)
                else:
                    # Fallback: try another sample
                    return self.__getitem__((index + 1) % len(self))
        else:
            # Original random sampling logic
            if curr_ts is None:
                # Sample a random curr_ts from the concatenated sequence
                try:
                    min_start = self.history_len * self.history_skip_frame
                    max_start = total_timesteps - self.prediction_offset - 1
                    if max_start < min_start:
                        # Not enough timesteps, try another sample
                        return self.__getitem__((index + 1) % len(self))
                    
                    # Randomly sample curr_ts from concatenated sequence
                    # Uses global random state: each call produces different random value
                    curr_ts = np.random.randint(min_start, max_start + 1)
                except ValueError:
                    return self.__getitem__((index + 1) % len(self))
        
        start_ts = curr_ts - self.history_len * self.history_skip_frame
        target_ts = curr_ts + self.prediction_offset
        
        # Determine which stage the target_ts belongs to
        target_stage_idx = self.stage_indices[-1]  # Default to last stage
        for i, (start_bound, end_bound) in enumerate(stage_boundaries):
            if start_bound <= target_ts < end_bound:
                target_stage_idx = self.stage_indices[i]
                break
        
        # Get command embedding and text for target stage
        # Note: target_stage_idx is 0-based internally, but stage_embeddings/stage_texts use 1-based keys
        stage_key = target_stage_idx + 1
        command_embedding = self.stage_embeddings.get(stage_key, None)
        if command_embedding is None:
            command_embedding = torch.zeros(768)
        else:
            command_embedding = torch.tensor(command_embedding).float().squeeze()
        
        command_gt = self.stage_texts.get(stage_key, f"stage_{target_stage_idx}")
        
        # Load image sequence for history frames
        # Need to map global timesteps to local timesteps within each run
        image_sequence = []
        sequence_replay = None
        for ts in range(start_ts, curr_ts + 1, self.history_skip_frame):
            # Find which stage/run this timestep belongs to
            run_idx = None
            local_ts = None
            for i, (start_bound, end_bound) in enumerate(stage_boundaries):
                if start_bound <= ts < end_bound:
                    run_idx = i
                    local_ts = ts - start_bound
                    break
            
            if run_idx is None:
                # If ts is beyond all stages, use the last run
                run_idx = len(selected_runs) - 1
                start_bound, end_bound = stage_boundaries[run_idx]
                # Clamp to the last run's valid range
                local_ts = min(ts - start_bound, run_lengths[run_idx] - 1)
            
            stage_idx, run_path = selected_runs[run_idx]
            
            # Load image for this timestep
            current_ts_images = []
            for cam_name in self.camera_names:
                cam_dir = os.path.join(run_path, cam_name)
                if not os.path.exists(cam_dir):
                    # Fallback: try another sample
                    return self.__getitem__((index + 1) % len(self))
                
                files = sorted([f for f in os.listdir(cam_dir) if f.endswith('_full.jpg')])
                if not files:
                    return self.__getitem__((index + 1) % len(self))
                
                # Use frame index corresponding to local_ts
                file_idx = min(local_ts, len(files) - 1)
                img_path = os.path.join(cam_dir, files[file_idx])
                
                # Load and convert image
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img)
                
                # Apply transformations
                if self.random_crop:
                    img_np = random_crop(img_np)
                
                current_ts_images.append(img_np)
            
            # Apply data augmentation if enabled (synchronized across cameras)
            if self.use_augmentation:
                if self.sync_sequence_augmentation:
                    current_ts_images, sequence_replay = self.image_preprocessor.augment_images(
                        current_ts_images,
                        self.training,
                        replay=sequence_replay,
                        return_replay=True,
                    )
                else:
                    current_ts_images = self.image_preprocessor.augment_images(
                        current_ts_images, self.training
                    )
            
            # Convert to tensor format (C, H, W) in [0, 1]
            current_ts_tensors = [self.to_tensor(img) for img in current_ts_images]
            all_cam_images = torch.stack(current_ts_tensors, dim=0)
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
        # Try to get stage_texts first, if not found, try stage_embeddings (which may contain text field)
        entries = data.get('stage_texts', None)
        if entries is None:
            entries = data.get('stage_embeddings', data)
        
        if isinstance(entries, list):
            stage_texts = {int(e['stage']): e['text'] for e in entries if 'stage' in e and 'text' in e}
        elif isinstance(entries, dict):
            # Only convert keys that can be converted to int (stage numbers)
            stage_texts = {}
            for k, v in entries.items():
                try:
                    stage_idx = int(k)
                    # v could be a string (text) or dict with 'text' field
                    if isinstance(v, str):
                        stage_texts[stage_idx] = v
                    elif isinstance(v, dict) and 'text' in v:
                        stage_texts[stage_idx] = v['text']
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


def load_composite_data(
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
    train_subdir: str | None = "training_datasets",
    val_subdir: str | None = "validation_datasets",
    val_root: str | None = None,  # Separate validation root directory
    use_augmentation=False,
    image_size=224,
    sampling_strategy='balanced',  # 'balanced' or 'random' or 'sequential'
    max_combinations=None,  # Maximum number of combinations (if None, use n_repeats)
    n_repeats=None,  # Number of times each run should be used (alternative to max_combinations)
    sync_sequence_augmentation: bool = False,
    use_weak_traversal: bool = False,  # Use weak traversal sampling strategy
    samples_non_cross_per_stage: int = 1,  # Number of non-cross-stage samples per stage (target_ts in current stage)
    samples_cross_per_stage: int = 1,  # Number of cross-stage samples per stage (target_ts in next stage)
):
    """
    Load composite dataset for HighLevelModel training.
    
    This function creates CompositeSequenceDataset instances that combine the advantages
    of SequenceDataset (complete sequences) and SplittedSequenceDataset (randomness).
    
    For each sample, it randomly selects one run from each stage, concatenates them into
    a complete sequence, and then samples from this concatenated sequence.
    
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
    # Use the same logic as load_splitted_data to ensure consistency
    # stage_texts keys should be 1-based (same as load_splitted_data)
    # CompositeSequenceDataset will use target_stage_idx (0-based) to look up, but will fall back to default
    stage_texts = {}
    if stage_texts_file and os.path.exists(stage_texts_file):
        with open(stage_texts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Try to get stage_texts first, if not found, try stage_embeddings (which may contain text field)
        entries = data.get('stage_texts', None)
        if entries is None:
            entries = data.get('stage_embeddings', data)
        
        if isinstance(entries, list):
            stage_texts = {int(e['stage']): e['text'] for e in entries if 'stage' in e and 'text' in e}
        elif isinstance(entries, dict):
            # Only convert keys that can be converted to int (stage numbers)
            # Keep keys as 1-based (same as load_splitted_data)
            stage_texts = {}
            for k, v in entries.items():
                try:
                    stage_idx = int(k)
                    # v could be a string (text) or dict with 'text' field
                    if isinstance(v, str):
                        stage_texts[stage_idx] = v
                    elif isinstance(v, dict) and 'text' in v:
                        stage_texts[stage_idx] = v['text']
                except (ValueError, TypeError):
                    # Skip non-numeric keys (like 'model', 'encoder', etc.)
                    continue
    
    # Determine train and validation root directories
    train_root_dir = os.path.join(root_dir, train_subdir) if train_subdir else root_dir
    
    # Handle validation directory (support separate val_root parameter)
    if val_root:
        val_root_dir = val_root
    elif val_subdir:
        val_root_dir = os.path.join(root_dir, val_subdir)
    else:
        val_root_dir = None
    
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
    train_dataset = CompositeSequenceDataset(
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
        sampling_strategy=sampling_strategy,
        max_combinations=max_combinations,
        n_repeats=n_repeats,
        sync_sequence_augmentation=sync_sequence_augmentation,
        use_weak_traversal=use_weak_traversal,
        samples_non_cross_per_stage=samples_non_cross_per_stage,
        samples_cross_per_stage=samples_cross_per_stage,
    )
    
    print(f"Training dataset: {len(train_dataset)} samples from {train_root_dir}")
    
    # Create validation dataset if validation directory exists
    val_dataset = None
    if val_root_dir and os.path.exists(val_root_dir):
        # Validation uses sequential strategy for deterministic evaluation
        val_dataset = CompositeSequenceDataset(
            val_root_dir,
            camera_names,
            history_len=history_len,
            prediction_offset=prediction_offset,
            history_skip_frame=history_skip_frame,
            random_crop=random_crop,
            stage_embeddings=stage_embeddings,
            stage_texts=stage_texts,
            use_augmentation=False,  # No augmentation for validation
            training=False,
            image_size=image_size,
            sampling_strategy='sequential',  # Always use sequential for validation
            max_combinations=None,  # Not used for sequential
            n_repeats=None,  # Not used for sequential
            sync_sequence_augmentation=False,
            use_weak_traversal=use_weak_traversal,  # Use same value for validation
            samples_non_cross_per_stage=samples_non_cross_per_stage,
        samples_cross_per_stage=samples_cross_per_stage,  # Use same value for validation
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
        image_sequence = []
        
        # Refactored to use synchronized augmentation
        for ts in hist_indices:
            current_ts_images = []
            for cam in self.camera_names:
                files_i = cam_to_files_i[cam]
                files_j = cam_to_files_j[cam]
                
                if ts < len_i:
                    path = files_i[min(ts, len_i - 1)]
                else:
                    idx = ts - len_i
                    path = files_j[min(idx, len_j - 1)]
                
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)
                current_ts_images.append(img_np)
            
            # Apply data augmentation if enabled (synchronized across cameras)
            if self.use_augmentation:
                current_ts_images = self.image_preprocessor.augment_images(
                    current_ts_images, training=True
                )
            
            # Convert to tensor format (C, H, W) in [0, 1]
            current_ts_tensors = [self.to_tensor(img) for img in current_ts_images]
            all_cam_images = torch.stack(current_ts_tensors, dim=0)
            image_sequence.append(all_cam_images)

        # Stack sequence: (T, K, C, H, W)
        image_sequence = torch.stack(image_sequence, dim=0)

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
        image_sequence = []
        
        # Refactored to use synchronized augmentation
        for ts in hist_indices:
            current_ts_images = []
            for cam in self.camera_names:
                # find which stage segment ts falls into
                seg_idx = int(np.searchsorted(cumulative, ts, side="right") - 1)
                seg_offset = ts - cumulative[seg_idx]
                files = cam_files_per_stage[seg_idx][cam]
                path = files[min(seg_offset, len(files) - 1)]
                
                img = Image.open(path).convert("RGB")
                img_np = np.array(img)
                current_ts_images.append(img_np)
            
            # Apply data augmentation if enabled (synchronized across cameras)
            if self.use_augmentation:
                current_ts_images = self.image_preprocessor.augment_images(
                    current_ts_images, training=True
                )
            
            # Convert to tensor format (C, H, W) in [0, 1]
            current_ts_tensors = [self.to_tensor(img) for img in current_ts_images]
            all_cam_images = torch.stack(current_ts_tensors, dim=0)
            image_sequence.append(all_cam_images)
            
        # Stack sequence: (T, K, C, H, W)
        image_sequence = torch.stack(image_sequence, dim=0)

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