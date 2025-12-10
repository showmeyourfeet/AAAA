"""Utility functions and dataset helpers for ACT."""

import json
import os
import random
from typing import Dict, Iterable, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# Albumentations is optional; it is only needed when augmentation is enabled.
try:
    import albumentations as A
except Exception:  # pragma: no cover - optional dependency
    A = None

# Optional image saving backends for debug outputs
try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional dependency
    PILImage = None
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency
    imageio = None


def compute_dict_mean(epoch_dicts: Sequence[Dict[str, torch.Tensor]]):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        if k == "epoch":
            # epoch should not be averaged, just take the first value (all should be the same)
            result[k] = epoch_dicts[0][k]
        else:
            value_sum = 0
            for epoch_dict in epoch_dicts:
                value_sum += epoch_dict[k]
            result[k] = value_sum / num_items
    return result


def detach_dict(d: Dict[str, torch.Tensor]):
    return {k: v.detach().cpu() for k, v in d.items()}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def is_multi_gpu_checkpoint(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("model.module.") for k in state_dict.keys())


def initialize_model_and_tokenizer(encoder: str):
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        DistilBertModel,
        DistilBertTokenizer,
    )

    if encoder == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif encoder == "clip":
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
    else:
        raise ValueError("Unknown encoder type. Please use 'distilbert' or 'clip'.")
    return tokenizer, model


def encode_text(text: Iterable[str] | str, encoder: str, tokenizer, model):
    if isinstance(text, str):
        batch = [text]
    else:
        batch = list(text)
        if len(batch) == 0:
            raise ValueError("encode_text received empty text iterable")
    if encoder == "distilbert":
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().tolist()
    if encoder == "clip":
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=77,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    raise ValueError(f"Unsupported encoder: {encoder}")


def crop_resize(
    image: np.ndarray,
    crop_h: int = 240,
    crop_w: int = 320,
    resize_h: int = 480,
    resize_w: int = 640,
    resize: bool = True,
) -> np.ndarray:
    h, w, _ = image.shape
    y1 = h - crop_h - 20
    x1 = (w - crop_w) // 2
    cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
    if resize:
        return cv2.resize(cropped, (resize_w, resize_h))
    return cropped


def random_crop(image: np.ndarray, crop_percentage: float = 0.95) -> np.ndarray:
    """
    Randomly crop an image by a percentage and resize back to original size.
    image: H x W x C (numpy, uint8 or float)
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop_percentage), int(w * crop_percentage)
    if new_h <= 0 or new_w <= 0:
        return image
    top = random.randint(0, max(0, h - new_h))
    left = random.randint(0, max(0, w - new_w))
    cropped = image[top : top + new_h, left : left + new_w, ...]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Dataset & data-loading utilities (ported from top-level utils.py)
# ---------------------------------------------------------------------------

# Cropping/Debug configuration (module-level; not stored on class)
# Set CROP_RANGE to a tuple (y1, y2, x1, x2) in pixels to enable cropping; None to disable.
CROP_RANGE = None  # e.g., (0, 400, 0, None)

# Enable saving cropped images as PNG for debugging.
DEBUG_SAVE_CROPS = False
# Directory to save debug images.
DEBUG_SAVE_DIR = "debug_crops"


# Dataset file integrity cache (module-level)
_DATASET_FILE_PARTITIONS: Dict[str, Dict[str, list]] = {}


def _extract_episode_id(filename: str) -> int | None:
    """Return the episode id encoded in an episode_<id>.hdf5 filename."""
    if not filename.startswith("episode_") or not filename.endswith(".hdf5"):
        return None
    try:
        return int(filename.split("_", 1)[1].rsplit(".", 1)[0])
    except (IndexError, ValueError):
        return None


def _scan_episode_files(dataset_dir: str) -> Dict[str, list]:
    """Scan dataset directory, caching valid and corrupted episode file paths."""
    valid_ids: list[int] = []
    valid_files: list[str] = []
    corrupted_files: list[str] = []
    corrupted_ids: list[int] = []

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    for fname in sorted(os.listdir(dataset_dir)):
        episode_id = _extract_episode_id(fname)
        if episode_id is None:
            continue
        fpath = os.path.join(dataset_dir, fname)
        try:
            with h5py.File(fpath, "r"):
                pass
        except (OSError, IOError):
            corrupted_files.append(fpath)
            corrupted_ids.append(episode_id)
            continue
        valid_ids.append(episode_id)
        valid_files.append(fpath)

    return {
        "valid_ids": valid_ids,
        "valid_files": valid_files,
        "corrupted_ids": corrupted_ids,
        "corrupted_files": corrupted_files,
    }


def get_episode_file_partitions(dataset_dir: str, refresh: bool = False) -> Dict[str, list]:
    """Return cached valid/corrupted episode file lists for a dataset directory."""
    cache = _DATASET_FILE_PARTITIONS.get(dataset_dir)
    if cache is None or refresh:
        cache = _scan_episode_files(dataset_dir)
        _DATASET_FILE_PARTITIONS[dataset_dir] = cache
        if cache["corrupted_files"]:
            print(
                f"[llp.utils] Detected {len(cache['corrupted_files'])} corrupted episode "
                f"files under {dataset_dir}; they will be skipped during training."
            )
    return cache


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        real_ids=None,
        is_cut: bool = False,
    ):
        super(EpisodicDataset, self).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.real_ids = real_ids
        self.is_cut = is_cut  # cut bottom of image or not
        if len(self.episode_ids) > 0:
            # initialize self.is_sim
            _ = self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        original_index = index
        sample_full_episode = False  # hardcode
        if self.real_ids is not None:
            episode_id = self.real_ids[original_index]
        else:
            episode_id = self.episode_ids[original_index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            # is_sim = root.attrs['sim']  # original format
            is_sim = True
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                if "/is_pad" in root:
                    pad_info = root["/is_pad"]
                    has_dataset_pad_info = True
                    num_valid_steps = episode_len - np.sum(pad_info)
                    start_ts = np.random.choice(int(num_valid_steps))
                else:
                    has_dataset_pad_info = False
                    start_ts = np.random.choice(episode_len)

            # get observation at start_ts only
            if "sim" in root.attrs:
                # original format
                qpos = root["/observations/qpos"][start_ts]
            else:
                # custom format: use lrstate
                qpos = root["/observations/lrstate"][start_ts]

            # get pad info from dataset
            if has_dataset_pad_info:
                is_pad_dataset = root["/is_pad"][start_ts:]

            image_dict = {}
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][start_ts]

            # get all actions after and including start_ts
            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                # hack, to make timesteps more aligned
                action = root["/action"][max(0, start_ts - 1) :]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1
        # union with dataset-provided pad info
        if "has_dataset_pad_info" in locals() and has_dataset_pad_info:
            is_pad[:action_len] = is_pad_dataset

        # stack multi-camera images -> (K, H, W, C)
        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)

        # optionally crop multi-camera images before converting to torch
        if CROP_RANGE is not None:
            all_cam_images = self._crop_all_cam_images(all_cam_images, CROP_RANGE)
            if DEBUG_SAVE_CROPS:
                self._save_crops_as_png(all_cam_images, episode_id, start_ts)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last -> channel first
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0

        # ensure norm stats are torch.float32 tensors to avoid dtype promotion
        if isinstance(self.norm_stats["action_mean"], np.ndarray):
            self.norm_stats["action_mean"] = torch.from_numpy(
                self.norm_stats["action_mean"]
            ).float()
        if isinstance(self.norm_stats["action_std"], np.ndarray):
            self.norm_stats["action_std"] = torch.from_numpy(
                self.norm_stats["action_std"]
            ).float()
        if isinstance(self.norm_stats["qpos_mean"], np.ndarray):
            self.norm_stats["qpos_mean"] = torch.from_numpy(
                self.norm_stats["qpos_mean"]
            ).float()
        if isinstance(self.norm_stats["qpos_std"], np.ndarray):
            self.norm_stats["qpos_std"] = torch.from_numpy(
                self.norm_stats["qpos_std"]
            ).float()

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad

    def _crop_all_cam_images(self, all_cam_images, crop_range):
        """Crop stacked multi-camera images with a given range.

        Args:
            all_cam_images: numpy array (K, H, W, C)
            crop_range: tuple (y1, y2, x1, x2) in pixels; supports None for edges
        """
        assert all_cam_images.ndim == 4, "Expected images with shape (K, H, W, C)"
        _, H, W, _ = all_cam_images.shape
        y1, y2, x1, x2 = crop_range

        # handle None and clip to bounds
        y1 = 0 if y1 is None else int(y1)
        y2 = H if y2 is None else int(y2)
        x1 = 0 if x1 is None else int(x1)
        x2 = W if x2 is None else int(x2)

        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))

        if not (y2 > y1 and x2 > x1):
            raise ValueError(
                f"Invalid crop_range {crop_range} for image size (H={H}, W={W})"
            )

        return all_cam_images[:, y1:y2, x1:x2, :]

    def _save_crops_as_png(self, images_khwc, episode_id, start_ts):
        """Save cropped images per camera as PNGs for debugging."""
        if PILImage is None and imageio is None and cv2 is None:
            return
        os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

        for i, cam_name in enumerate(self.camera_names):
            img = images_khwc[i]
            fname = os.path.join(
                DEBUG_SAVE_DIR, f"ep{episode_id}_t{start_ts}_{cam_name}.png"
            )

            # Try PIL first
            saved = False
            if PILImage is not None:
                try:
                    PILImage.fromarray(img.astype(np.uint8)).save(fname)
                    saved = True
                except Exception:
                    saved = False
            if not saved and imageio is not None:
                try:
                    imageio.imwrite(fname, img.astype(np.uint8))
                    saved = True
                except Exception:
                    saved = False
            if not saved and cv2 is not None:
                try:
                    # cv2 expects BGR
                    if img.shape[-1] == 3:
                        cv2.imwrite(
                            fname, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        )
                    else:
                        cv2.imwrite(fname, img.astype(np.uint8))
                    saved = True
                except Exception:
                    saved = False


def get_norm_stats(dataset_dir, num_episodes, real_ids=None):
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        if real_ids is not None:
            episode_idx = real_ids[episode_idx]
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            if "sim" in root.attrs:
                # original format
                qpos = root["/observations/qpos"][()]
                action = root["/action"][()]
            else:
                # custom format
                qpos = root["/observations/lrstate"][()]
                action = root["/action"][()]
                _ = root["/is_pad"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data).float()
    all_action_data = torch.stack(all_action_data).float()

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    print(
        f"action mean: {action_mean}, action std: {action_std} \n "
        f"qpos mean: {qpos_mean}, qpos std: {qpos_std}"
    )
    # return stats as torch.float32 tensors to keep dtype consistent downstream
    stats = {
        "action_mean": action_mean.squeeze().float(),
        "action_std": action_std.squeeze().float(),
        "qpos_mean": qpos_mean.squeeze().float(),
        "qpos_std": qpos_std.squeeze().float(),
        "example_qpos": torch.from_numpy(qpos).float(),
    }

    return stats


def get_norm_stats_with_valid_len(dataset_dir, num_episodes, real_ids=None):
    all_qpos_data = []
    all_action_data = []

    for episode_idx in range(num_episodes):
        if real_ids is not None:
            episode_idx = real_ids[episode_idx]
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            if "sim" in root.attrs:
                # original format
                qpos = root["/observations/qpos"][()]
                action = root["/action"][()]
            else:
                # custom format
                qpos = root["/observations/lrstate"][()]
                action = root["/action"][()]
                if "/is_pad" in root:
                    is_pad = root["/is_pad"][()]
                    valid_len = int(qpos.shape[0] - np.sum(is_pad))
                    qpos = qpos[:valid_len]
                    action = action[:valid_len]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    print(
        f"action mean: {action_mean}, action std: {action_std} \n "
        f"qpos mean: {qpos_mean}, qpos std: {qpos_std}"
    )
    # return stats as torch.float32 tensors to keep dtype consistent downstream
    stats = {
        "action_mean": action_mean.squeeze().float(),
        "action_std": action_std.squeeze().float(),
        "qpos_mean": qpos_mean.squeeze().float(),
        "qpos_std": qpos_std.squeeze().float(),
        "example_qpos": torch.from_numpy(qpos).float(),
    }

    return stats


def load_data(
    dataset_dir, 
    num_episodes, 
    camera_names, 
    batch_size_train, 
    batch_size_val, 
    train_ratio=0.8,
    prefetch_factor=1,
    num_workers=1,
):
    print(f"\nData from: {dataset_dir}\n")
    partitions = get_episode_file_partitions(dataset_dir)
    real_ids = partitions["valid_ids"]
    num_available = len(real_ids)
    if num_available == 0:
        raise RuntimeError(
            f"No valid episode files found in {dataset_dir}. "
            "Check dataset integrity."
        )
    if num_episodes != num_available:
        print(
            f"[llp.utils] Requested {num_episodes} episodes but only "
            f"{num_available} valid episodes are available; proceeding with the "
            "valid subset."
        )
    num_episodes = num_available
    # obtain train test split
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]
    real_train_ids = [real_ids[i] for i in train_indices]
    real_val_ids = [real_ids[i] for i in val_indices]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_with_valid_len(
        dataset_dir, num_episodes, real_ids=real_ids
    )

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        real_ids=real_train_ids,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        real_ids=real_val_ids,
    )
    
    # When num_workers=0, prefetch_factor must be None
    train_prefetch_factor = None if num_workers == 0 else prefetch_factor
    val_num_workers = int(num_workers * 0.2) if num_workers > 0 else 0
    val_prefetch_factor = None if val_num_workers == 0 else int(prefetch_factor * 0.2)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=train_prefetch_factor,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=val_num_workers,
        prefetch_factor=val_prefetch_factor,
    )

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def build_default_albu_augmentations(
    height,
    width,
    use_replay: bool = True,
    crop_scale=(0.8, 1.0),
    crop_ratio=(0.9, 1.1),
    rotate_limit=15,
    shift_limit=0.0625,
    scale_limit=0.1,
    jitter_brightness=0.2,
    jitter_contrast=0.2,
    jitter_saturation=0.2,
    jitter_hue=0.1,
    coarse_max_holes=8,
    coarse_min_holes=1,
    coarse_max_size=0.1,
    coarse_min_size=0.05,
    p_color=0.8,
    p_coarse=0.5,
):
    """Build a default Albumentations pipeline for images."""
    if A is None:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install via `pip install albumentations`."
        )

    def to_pixels(val, dim):
        if 0 < val <= 1:
            return max(1, int(round(val * dim)))
        return int(val)

    max_h = to_pixels(coarse_max_size, height)
    max_w = to_pixels(coarse_max_size, width)
    min_h = to_pixels(coarse_min_size, height)
    min_w = to_pixels(coarse_min_size, width)

    tfs = [
        A.RandomResizedCrop(
            height=height, width=width, scale=crop_scale, ratio=crop_ratio, p=1.0
        ),
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            p=0.8,
        ),
        A.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
            hue=jitter_hue,
            p=p_color,
        ),
        A.CoarseDropout(
            max_holes=coarse_max_holes,
            min_holes=coarse_min_holes,
            max_height=max_h,
            max_width=max_w,
            min_height=min_h,
            min_width=min_w,
            p=p_coarse,
        ),
    ]

    return A.ReplayCompose(tfs) if use_replay else A.Compose(tfs)


def apply_albu_augmentations_multicam(
    images_khwc, build_if_none, cached_augmentor_ref=None, same_on_all: bool = True
):
    """Apply Albumentations augmentation to multi-camera numpy images."""
    if A is None:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install via `pip install albumentations`."
        )

    assert images_khwc.ndim == 4, "Expected images shape (K, H, W, C)"
    K, H, W, C = images_khwc.shape
    _ = (K, W, C)  # unused vars but keep for clarity

    # Build and/or fetch cached augmentor
    augmentor = None
    if cached_augmentor_ref is not None:
        augmentor = getattr(cached_augmentor_ref, "_augmentor", None)
        if augmentor is None:
            augmentor = build_if_none(H, W)
            setattr(cached_augmentor_ref, "_augmentor", augmentor)
    else:
        augmentor = build_if_none(H, W)

    out = []
    if same_on_all and isinstance(augmentor, A.ReplayCompose):
        # Sample params on first image, then replay for others
        res0 = augmentor(image=images_khwc[0])
        out.append(res0["image"])  # ReplayCompose returns dict with replay
        replay = res0.get("replay", None)
        for i in range(1, K):
            out.append(A.ReplayCompose.replay(replay, image=images_khwc[i])["image"])
    else:
        # Independent augmentation for each camera image
        for i in range(K):
            out.append(augmentor(image=images_khwc[i])["image"])

    return np.stack(out, axis=0)