"""Dataset classes and data loading utilities for ACT."""

import json
import os
import random
import re
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
import torchvision.transforms as T

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None


class MixedRatioSampler(Sampler):
    """
    Sampler that mixes samples from two datasets according to a ratio.
    
    Args:
        normal_dataset: Normal dataset
        dagger_dataset: DAgger dataset (correction command data)
        mix_ratio: Ratio of dagger samples in each batch (0.0-1.0)
        batch_size: Batch size for sampling
        shuffle: Whether to shuffle samples
    """
    def __init__(self, normal_dataset, dagger_dataset, mix_ratio=0.5, batch_size=1, shuffle=True):
        self.normal_dataset = normal_dataset
        self.dagger_dataset = dagger_dataset
        self.mix_ratio = mix_ratio
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.normal_size = len(normal_dataset)
        self.dagger_size = len(dagger_dataset)
        
        # Calculate number of samples per batch from each dataset
        self.dagger_per_batch = max(1, int(batch_size * mix_ratio))
        self.normal_per_batch = batch_size - self.dagger_per_batch
        
        # Create indices for both datasets
        self.normal_indices = list(range(self.normal_size))
        self.dagger_indices = list(range(self.dagger_size))
        
        # Calculate total number of batches
        # We need enough samples from both datasets
        normal_batches = self.normal_size // self.normal_per_batch if self.normal_per_batch > 0 else 0
        dagger_batches = self.dagger_size // self.dagger_per_batch if self.dagger_per_batch > 0 else 0
        self.num_batches = min(normal_batches, dagger_batches) if normal_batches > 0 and dagger_batches > 0 else max(normal_batches, dagger_batches)
        
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.normal_indices)
            random.shuffle(self.dagger_indices)
        
        normal_iter = iter(self.normal_indices)
        dagger_iter = iter(self.dagger_indices)
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            # Add normal samples
            for _ in range(self.normal_per_batch):
                try:
                    idx = next(normal_iter)
                    batch_indices.append(('normal', idx))
                except StopIteration:
                    # Recycle normal indices if exhausted
                    normal_iter = iter(random.sample(self.normal_indices, len(self.normal_indices)) if self.shuffle else self.normal_indices)
                    idx = next(normal_iter)
                    batch_indices.append(('normal', idx))
            
            # Add dagger samples
            for _ in range(self.dagger_per_batch):
                try:
                    idx = next(dagger_iter)
                    batch_indices.append(('dagger', idx))
                except StopIteration:
                    # Recycle dagger indices if exhausted
                    dagger_iter = iter(random.sample(self.dagger_indices, len(self.dagger_indices)) if self.shuffle else self.dagger_indices)
                    idx = next(dagger_iter)
                    batch_indices.append(('dagger', idx))
            
            # Shuffle batch to mix normal and dagger samples
            if self.shuffle:
                random.shuffle(batch_indices)
            
            yield batch_indices
    
    def __len__(self):
        return self.num_batches


class MixedDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that combines normal and dagger datasets.
    Works with MixedRatioSampler which yields batches of (dataset_type, idx) tuples.
    """
    def __init__(self, normal_dataset, dagger_dataset):
        self.normal_dataset = normal_dataset
        self.dagger_dataset = dagger_dataset
    
    def __getitem__(self, item):
        # item is a tuple (dataset_type, idx) from MixedRatioSampler
        if isinstance(item, tuple) and len(item) == 2:
            dataset_type, idx = item
            if dataset_type == 'normal':
                return self.normal_dataset[idx]
            else:
                return self.dagger_dataset[idx]
        else:
            # Fallback: treat as normal index (for compatibility)
            if item < len(self.normal_dataset):
                return self.normal_dataset[item]
            else:
                return self.dagger_dataset[item - len(self.normal_dataset)]
    
    def __len__(self):
        return len(self.normal_dataset) + len(self.dagger_dataset)


def _infer_state_dim_from_stats(norm_stats: Dict[str, np.ndarray]) -> int:
    """Infer qpos/state dimension from normalization stats."""
    for key in ("example_qpos", "qpos_mean", "state_mean"):
        value = norm_stats.get(key)
        if value is not None:
            arr = np.asarray(value).reshape(-1)
            return int(arr.shape[0])
    action_mean = norm_stats.get("action_mean")
    if action_mean is not None:
        return int(np.asarray(action_mean).reshape(-1).shape[0])
    raise ValueError("Unable to infer state dimension from normalization stats.")


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
            # Color-only augmentation to avoid geometric mismatch between cameras.
            self.albumentations_transform = A.ReplayCompose(
                [
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2, contrast_limit=0.2, p=0.8
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=20,
                                sat_shift_limit=30,
                                val_shift_limit=20,
                                p=0.8,
                            ),
                            A.ColorJitter(
                                brightness=0.2,
                                contrast=0.2,
                                saturation=0.2,
                                hue=0.1,
                                p=0.8,
                            ),
                        ],
                        p=0.9,
                    )
                ]
            )
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
        """Apply augmentation to a single numpy array image."""
        augmented = self.augment_images([image_np], training)
        return augmented[0]

    def augment_images(self, images: list[np.ndarray], training: bool) -> list[np.ndarray]:
        """Apply consistent augmentation with replay support across multiple cameras."""
        if (
            not training
            or not self.use_augmentation
            or self.albumentations_transform is None
            or not images
        ):
            return images

        augmented0 = self.albumentations_transform(image=images[0])
        replay = augmented0.get("replay")
        if replay is None:
            # Fallback to independent augmentation if replay is unexpectedly missing.
            return [augmented0["image"]] + [
                self.albumentations_transform(image=img)["image"] for img in images[1:]
            ]

        results = [augmented0["image"]]
        for img in images[1:]:
            augmented = A.ReplayCompose.replay(replay, image=img)
            results.append(augmented["image"])
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


class SplittedEpisodicDataset(torch.utils.data.Dataset):
    """Dataset adapter for stage/run-style datasets.
    
    Supports two sampling modes:
    1. Single random (default): Traverse all stage/run combinations, randomly sample start_ts
    2. Episodic sampling: Traverse run IDs (episodes), randomly sample stage (segment), then start_ts
    """

    def __init__(
        self,
        root_dir: str,
        camera_names: Sequence[str],
        norm_stats: Dict[str, np.ndarray],
        max_len: int | None = None,
        use_language: bool = False,
        stage_embeddings: Dict[int, Sequence[float]] | None = None,
        correction_embeddings: Dict[int, Sequence[float]] | None = None,
        image_size: int = 224,
        use_augmentation: bool = False,
        training: bool = True,
        use_episodic_sampling: bool = False,
        use_state: bool = True,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.max_len = max_len
        self.use_language = use_language
        self.stage_embeddings = stage_embeddings or {}
        self.correction_embeddings = correction_embeddings or {}
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.training = training
        self.use_episodic_sampling = use_episodic_sampling
        self.use_state = use_state
        self.state_dim = _infer_state_dim_from_stats(norm_stats)
        self._zero_state = torch.zeros(self.state_dim, dtype=torch.float32)
        
        # Initialize ImagePreprocessor if augmentation is enabled
        if self.use_augmentation:
            self.image_preprocessor = ImagePreprocessor(
                image_size=image_size,
                use_augmentation=use_augmentation
            )

        # Collect all runs
        all_runs: list[Tuple[int, str]] = []
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
                    all_runs.append((stage_idx, run_path))

        if len(all_runs) == 0:
            print(f"[SplittedEpisodicDataset] No runs found under {root_dir}")

        if self.use_episodic_sampling:
            # Group runs by run ID (episode level)
            # Extract run_id from run_path, e.g., "stage1/run_001" -> run_id = 1
            self.runs_by_episode: Dict[int, list[Tuple[int, str]]] = {}
            for stage_idx, run_path in all_runs:
                # Extract run number from path like "stage1/run_001" or "stage1/run001"
                run_dir = os.path.basename(run_path)
                # Try to extract number from "run_001", "run001", "run_1", "run1", etc.
                match = re.search(r"run[_\s]*(\d+)", run_dir, re.IGNORECASE)
                if match:
                    run_id = int(match.group(1))
                    if run_id not in self.runs_by_episode:
                        self.runs_by_episode[run_id] = []
                    self.runs_by_episode[run_id].append((stage_idx, run_path))
            
            # Sort by stage_idx within each episode
            for run_id in self.runs_by_episode:
                self.runs_by_episode[run_id].sort(key=lambda x: x[0])
            
            # Episode IDs are the run IDs
            self.episode_ids = sorted(self.runs_by_episode.keys())
            if len(self.episode_ids) == 0:
                print(f"[SplittedEpisodicDataset] Warning: No valid run IDs found. Falling back to single random mode.")
                self.use_episodic_sampling = False
                self.runs = all_runs
            else:
                print(f"[SplittedEpisodicDataset] Episodic sampling mode: {len(self.episode_ids)} episodes (run IDs)")
        else:
            # Single random mode: flat list of all stage/run combinations
            self.runs = all_runs
            self.runs_by_episode = None
            self.episode_ids = None

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        if self.use_episodic_sampling:
            return len(self.episode_ids)
        else:
            return len(self.runs)

    def _parse_run(self, run_path: str):
        """Parse data.txt to extract labels (stages), actions, qpos, and correction info.
        
        Correction info is parsed from the file header "Correction Info:" JSON section,
        which contains ranges of frames that use correction commands.
        """
        data_file = os.path.join(run_path, "data.txt")
        with open(data_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse Correction Info from file header (for DAgger datasets)
        correction_segments = []
        correction_info_match = re.search(r'Correction Info:\s*\n(\[.*?\])', content, re.DOTALL)
        if correction_info_match:
            try:
                correction_json = correction_info_match.group(1)
                correction_segments = json.loads(correction_json)
            except Exception as e:
                print(f"Warning: Failed to parse Correction Info in {data_file}: {e}")

        # Extract stage from file header (preprocessed format has stage info at the top)
        stage_match = re.search(r'stage\s*:\s*(\d+)', content)
        if stage_match:
            stage_num = int(stage_match.group(1)) - 1  # stage1 -> 0
        else:
            # Fallback: try to extract from "Stage: X" line
            stage_match = re.search(r'^Stage:\s*(\d+)', content, re.MULTILINE)
            if stage_match:
                stage_num = int(stage_match.group(1)) - 1
            else:
                stage_num = 0  # Default to stage 0 if not found

        labels: list[int] = []
        actions: list[np.ndarray] = []
        qpos: list[np.ndarray] = []

        frame_blocks = re.split(r"Frame_\d+:", content)[1:]
        for frame_idx, block in enumerate(frame_blocks):
            # Stage info is in file header, not in each frame block (preprocessed format)
            m_act = re.search(r"action:\s*\[([^\]]+)\]", block)
            m_lr = re.search(r"lrstate\s*:\s*\[([^\]]+)\]", block)
            
            if m_act and m_lr:
                try:
                    labels.append(stage_num)  # Use stage from file header
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
        
        if not actions:
            return None, None, None, None, None
        
        # Build correction_flags and correction_command_indices arrays based on Correction Info
        num_frames = len(actions)
        correction_flags = np.zeros(num_frames, dtype=np.float32)
        correction_command_indices = np.zeros(num_frames, dtype=np.int64) - 1  # -1 means no correction
        
        for segment in correction_segments:
            start = segment.get('correction_flag_start', 1) - 1  # Convert to 0-indexed
            end = segment.get('correction_flag_end', 1) - 1  # Convert to 0-indexed
            command_idx = segment.get('correction_command_idx', -1)
            
            # Set correction flags for the segment range (inclusive)
            for i in range(max(0, start), min(num_frames, end + 1)):
                correction_flags[i] = 1.0
                if command_idx >= 0:
                    correction_command_indices[i] = command_idx
        
        return (
            labels,
            np.stack(actions) if actions else None,
            np.stack(qpos) if qpos else None,
            correction_flags,
            correction_command_indices,
        )

    def __getitem__(self, index: int):
        if self.use_episodic_sampling:
            # Episodic sampling mode: similar to EpisodicDataset
            # 1. Get episode (run_id) from index
            run_id = self.episode_ids[index]
            # 2. Randomly select a stage (segment) from this episode
            available_stages = self.runs_by_episode[run_id]
            if len(available_stages) == 0:
                raise ValueError(f"No stages found for run_id {run_id}")
            stage_idx, run_path = random.choice(available_stages)
        else:
            # Single random mode: directly get stage/run from index
            stage_idx, run_path = self.runs[index]
        
        parse_result = self._parse_run(run_path)
        if parse_result[1] is None:
            # Fallback: return empty data
            act_dim = self.norm_stats["action_mean"].shape[0]
            qpos_dim = self.norm_stats["qpos_mean"].shape[0]
            actions_np = np.zeros((self.max_len, act_dim), dtype=np.float32)
            qpos_np = np.zeros((self.max_len, qpos_dim), dtype=np.float32)
            T = self.max_len
            correction_flags = np.zeros(T, dtype=np.float32)
            correction_command_indices = np.zeros(T, dtype=np.int64) - 1
        else:
            labels, actions_np, qpos_np, correction_flags, correction_command_indices = parse_result
            T = actions_np.shape[0]
            
            # Handle None correction info (for normal datasets without correction info)
            if correction_flags is None:
                correction_flags = np.zeros(T, dtype=np.float32)
            if correction_command_indices is None:
                correction_command_indices = np.zeros(T, dtype=np.int64) - 1
            
            # Handle None correction info (for normal datasets without correction info)
            if correction_flags is None:
                correction_flags = np.zeros(T, dtype=np.float32)
            if correction_command_indices is None:
                correction_command_indices = np.zeros(T, dtype=np.int64) - 1

        # 3. Randomly select start_ts within the selected stage/run
        start_ts = np.random.randint(0, T)
        chunk_len = min(self.max_len, T - start_ts)
        end_ts = start_ts + max(chunk_len - 1, 0)
        pad_length = self.max_len

        if self.use_state and qpos_np is not None:
            qpos = qpos_np[start_ts]
        else:
            qpos = None
        
        # Determine if current frame uses correction command based on Correction Info ranges
        is_correction = False
        correction_command_idx = -1
        if start_ts < len(correction_flags):
            is_correction = bool(correction_flags[start_ts])
            if start_ts < len(correction_command_indices):
                correction_command_idx = int(correction_command_indices[start_ts])

        current_images = []
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
            current_images.append(np.array(img))

        if self.use_augmentation and current_images:
            current_images = self.image_preprocessor.augment_images(
                current_images, self.training
            )

        all_cam_images = [self.to_tensor(img_np) for img_np in current_images]

        image_data = torch.stack(all_cam_images, dim=0)

        action_slice = actions_np[start_ts : end_ts + 1]
        action_len = action_slice.shape[0]
        padded_action = np.zeros((pad_length, action_slice.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action_slice
        is_pad = np.zeros(pad_length, dtype=np.bool_)
        is_pad[action_len:] = True

        action_data = torch.from_numpy(padded_action).float()
        action_data = (
            action_data - torch.from_numpy(self.norm_stats["action_mean"]).float()
        ) / torch.from_numpy(self.norm_stats["action_std"]).float()

        if self.use_state and qpos is not None:
            qpos_data = torch.from_numpy(qpos).float()
            qpos_data = (
                qpos_data - torch.from_numpy(self.norm_stats["qpos_mean"]).float()
            ) / torch.from_numpy(self.norm_stats["qpos_std"]).float()
        else:
            qpos_data = self._zero_state.clone()
        is_pad = torch.from_numpy(is_pad).bool()

        if self.use_language:
            # Select embedding based on correction_flag:
            # - If correction_flag=1 and correction_command_idx is valid: use correction embedding
            # - Otherwise: use instruction (stage) embedding
            if is_correction and correction_command_idx >= 0 and correction_command_idx in self.correction_embeddings:
                emb = self.correction_embeddings[correction_command_idx]
                if emb is None:
                    # Fallback to stage embedding if correction embedding not found
                    emb = self.stage_embeddings.get(stage_idx, None)
                    if emb is None:
                        emb = torch.zeros(768)
                    else:
                        emb = torch.tensor(emb).float()
                else:
                    emb = torch.tensor(emb).float()
            else:
                # Use instruction (stage) embedding
                emb = self.stage_embeddings.get(stage_idx, None)
                if emb is None:
                    emb = torch.zeros(768)
                else:
                    emb = torch.tensor(emb).float()
            return image_data, qpos_data, action_data, is_pad, emb
        return image_data, qpos_data, action_data, is_pad


def load_splitted_data(
    root_dir: str,
    camera_names: Sequence[str],
    batch_size_train: int,
    max_len: int,
    norm_stats: Dict[str, np.ndarray],
    use_language: bool = False,
    stage_embeddings_file: str | None = None,
    dagger_embeddings_file: str | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
    use_episodic_sampling: bool = False,
    use_state: bool = True,
    prefetch_factor: int | None = 2,
    num_workers: int = 8,
):
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
    
    correction_embeddings = None
    if use_language and dagger_embeddings_file and os.path.exists(dagger_embeddings_file):
        with open(dagger_embeddings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("correction_embeddings", data)
        if isinstance(entries, list):
            # List format: [{correction_command_idx: 1, embedding: [...], ...}, ...]
            correction_embeddings = {
                int(e["correction_command_idx"]): e["embedding"]
                for e in entries
                if "correction_command_idx" in e and "embedding" in e
            }
        elif isinstance(entries, dict):
            # Dict format: {"1": {"embedding": [...]}, ...}
            correction_embeddings = {
                int(k): v.get("embedding") if isinstance(v, dict) else v
                for k, v in entries.items()
                if "embedding" in (v if isinstance(v, dict) else {})
            }

    dataset = SplittedEpisodicDataset(
        root_dir,
        camera_names,
        norm_stats,
        max_len=max_len,
        use_language=use_language,
        stage_embeddings=stage_embeddings,
        correction_embeddings=correction_embeddings,
        image_size=image_size,
        use_augmentation=use_augmentation,
        training=True,
        use_episodic_sampling=use_episodic_sampling,
        use_state=use_state,
    )
    effective_prefetch = prefetch_factor if (num_workers and num_workers > 0) else None

    # Worker init function to set random seed for each worker process
    # This ensures reproducibility when using multiple workers
    def worker_init_fn(worker_id):
        import random
        import numpy as np
        import torch
        # Use a combination of base seed and worker_id to ensure each worker has different but deterministic random state
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=effective_prefetch,
        persistent_workers=False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    return train_dataloader, norm_stats, False


def load_splitted_data_with_dagger(
    normal_root_dir: str,
    dagger_root_dir: str,
    camera_names: Sequence[str],
    batch_size_train: int,
    max_len: int,
    norm_stats: Dict[str, np.ndarray],
    mix_ratio: float = 0.5,
    use_language: bool = False,
    stage_embeddings_file: str | None = None,
    dagger_embeddings_file: str | None = None,
    use_augmentation: bool = False,
    image_size: int = 224,
    use_episodic_sampling: bool = False,
    use_state: bool = True,
    prefetch_factor: int | None = 2,
    num_workers: int = 8,
):
    """
    Load splitted data with DAgger support: mix normal and correction command data.
    
    Args:
        normal_root_dir: Root directory for normal training datasets
        dagger_root_dir: Root directory for DAgger training datasets
        mix_ratio: Ratio of dagger samples in each batch (0.0-1.0)
        dagger_embeddings_file: JSON file containing correction command embeddings
        ... (other args same as load_splitted_data)
    
    Returns:
        train_dataloader: Mixed DataLoader with normal and dagger samples
        norm_stats: Normalization statistics
        False: (legacy return value)
    """
    assert 0.0 <= mix_ratio <= 1.0, "mix_ratio must be between 0.0 and 1.0"
    
    stage_embeddings = None
    if use_language and stage_embeddings_file and os.path.exists(stage_embeddings_file):
        with open(stage_embeddings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("stage_embeddings", data)
        stage_embeddings = {
            int(e["stage"]) - 1: e["embedding"]
            for e in entries
            if "stage" in e and "embedding" in e
        }
    
    correction_embeddings = None
    if use_language and dagger_embeddings_file and os.path.exists(dagger_embeddings_file):
        with open(dagger_embeddings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("correction_embeddings", data)
        if isinstance(entries, list):
            # List format: [{correction_command_idx: 1, embedding: [...], ...}, ...]
            correction_embeddings = {
                int(e["correction_command_idx"]): e["embedding"]
                for e in entries
                if "correction_command_idx" in e and "embedding" in e
            }
        elif isinstance(entries, dict):
            # Dict format: {"1": {"embedding": [...]}, ...}
            correction_embeddings = {
                int(k): v.get("embedding") if isinstance(v, dict) else v
                for k, v in entries.items()
                if "embedding" in (v if isinstance(v, dict) else {})
            }
    
    # Create normal dataset (no correction embeddings needed)
    normal_dataset = SplittedEpisodicDataset(
        normal_root_dir,
        camera_names,
        norm_stats,
        max_len=max_len,
        use_language=use_language,
        stage_embeddings=stage_embeddings,
        correction_embeddings=None,  # Normal data doesn't use correction embeddings
        image_size=image_size,
        use_augmentation=use_augmentation,
        training=True,
        use_episodic_sampling=use_episodic_sampling,
        use_state=use_state,
    )
    
    # Create dagger dataset (with correction embeddings support)
    dagger_dataset = SplittedEpisodicDataset(
        dagger_root_dir,
        camera_names,
        norm_stats,
        max_len=max_len,
        use_language=use_language,
        stage_embeddings=stage_embeddings,
        correction_embeddings=correction_embeddings,  # DAgger data uses correction embeddings
        image_size=image_size,
        use_augmentation=use_augmentation,
        training=True,
        use_episodic_sampling=use_episodic_sampling,
        use_state=use_state,
    )
    
    # Create mixed dataset
    mixed_dataset = MixedDataset(normal_dataset, dagger_dataset)
    
    # Create mixed sampler
    mixed_sampler = MixedRatioSampler(
        normal_dataset=normal_dataset,
        dagger_dataset=dagger_dataset,
        mix_ratio=mix_ratio,
        batch_size=batch_size_train,
        shuffle=True,
    )
    
    effective_prefetch = prefetch_factor if (num_workers and num_workers > 0) else None
    
    def worker_init_fn(worker_id):
        import random
        import numpy as np
        import torch
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_dataloader = DataLoader(
        mixed_dataset,
        batch_sampler=mixed_sampler,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=effective_prefetch,
        persistent_workers=False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    
    print(f"Loaded DAgger datasets:")
    print(f"  Normal dataset: {len(normal_dataset)} samples")
    print(f"  DAgger dataset: {len(dagger_dataset)} samples")
    print(f"  Mix ratio: {mix_ratio} (dagger samples per batch: {mixed_sampler.dagger_per_batch}/{batch_size_train})")
    
    return train_dataloader, norm_stats, False

