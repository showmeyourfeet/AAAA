import numpy as np
import torch
import os
import json
import re
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from typing import Sequence
import albumentations as A

from hlp.utils import random_crop


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
        normal_root_dir: str | None = None,  # Optional: additional root for normal data (for DAgger mixing)
        dagger_root_dir: str | None = None,  # Optional: additional root for dagger data (for DAgger mixing)
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
        # Support mixing runs from multiple root directories (for DAgger)
        self.runs_by_stage = {}
        
        # Collect runs from all root directories
        root_dirs = []
        # Add primary root_dir if it exists
        if root_dir and os.path.exists(root_dir):
            root_dirs.append(root_dir)
        # Add normal_root_dir if provided and different from root_dir
        if normal_root_dir and os.path.exists(normal_root_dir) and normal_root_dir != root_dir:
            root_dirs.append(normal_root_dir)
        # Add dagger_root_dir if provided and different from root_dir
        if dagger_root_dir and os.path.exists(dagger_root_dir) and dagger_root_dir != root_dir:
            root_dirs.append(dagger_root_dir)
        
        for root in root_dirs:
            for stage_dir in sorted(os.listdir(root)):
                stage_path = os.path.join(root, stage_dir)
                if not (stage_dir.startswith("stage") and os.path.isdir(stage_path)):
                    continue
                try:
                    stage_idx = int(stage_dir[5:]) - 1  # stage1 -> 0, stage2 -> 1, etc.
                except Exception:
                    continue
                
                if stage_idx not in self.runs_by_stage:
                    self.runs_by_stage[stage_idx] = []
                
                for run_dir in sorted(os.listdir(stage_path)):
                    run_path = os.path.join(stage_path, run_dir)
                    if run_dir.startswith("run") and os.path.isdir(run_path):
                        self.runs_by_stage[stage_idx].append(run_path)
        
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
                parse_result = self._parse_run(selected_run)
                if parse_result is None or parse_result[1] is None or len(parse_result[1]) < 1:
                    # Skip this combination if we can't parse it
                    break
                # _parse_run returns (labels, actions, qpos, correction_flags, correction_command_indices)
                # For weak traversal precomputation, we only need actions to get run length
                labels, actions_np, qpos_np = parse_result[0], parse_result[1], parse_result[2]
                
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
                # target_ts = curr_ts + prediction_offset should be in [stage_start, stage_end)
                # So curr_ts should be in [stage_start - prediction_offset, stage_end - prediction_offset)
                non_cross_min_curr = max(min_start, stage_start - self.prediction_offset)
                non_cross_max_curr = min(stage_end - self.prediction_offset - 1, max_start)
                
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
        """Parse data.txt to extract labels (stages), actions, qpos, and correction info."""
        data_file = os.path.join(run_path, 'data.txt')
        if not os.path.exists(data_file):
            return None, None, None, None, None
        
        with open(data_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse Correction Info if present (for DAgger datasets)
        correction_segments = []
        correction_info_match = re.search(r'Correction Info:\s*\n(\[.*?\])', content, re.DOTALL)
        if correction_info_match:
            try:
                import json
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
        
        labels, actions, qpos = [], [], []
        frame_blocks = re.split(r'Frame_\d+:', content)[1:]
        
        for frame_idx, b in enumerate(frame_blocks):
            # Stage info is in file header, not in each frame block (preprocessed format)
            m_act = re.search(r'action:\s*\[([^\]]+)\]', b)
            m_lr = re.search(r'lrstate\s*:\s*\[([^\]]+)\]', b)
            if m_act and m_lr:
                try:
                    labels.append(stage_num)  # Use stage from file header
                    actions.append(np.array([float(x.strip()) for x in m_act.group(1).split(',')], dtype=np.float32))
                    qpos.append(np.array([float(x.strip()) for x in m_lr.group(1).split(',')], dtype=np.float32))
                except Exception:
                    continue
        
        if not actions:
            return None, None, None, None, None
        
        # Build correction_flags and correction_command_indices arrays
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
        
        return labels, np.stack(actions), np.stack(qpos), correction_flags, correction_command_indices
    
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
        all_correction_flags = []  # Store correction flags for all runs
        all_correction_command_indices = []  # Store correction command indices for all runs
        
        for stage_idx, selected_run in selected_runs:
            if selected_run is None:
                # Fallback for any issues
                return self.__getitem__((index + 1) % len(self))
            
            # Parse run to get its length and correction info
            parse_result = self._parse_run(selected_run)
            if parse_result[1] is None or len(parse_result[1]) < 1:
                # Fallback: try another sample
                return self.__getitem__((index + 1) % len(self))
            
            labels, actions_np, qpos_np, correction_flags, correction_command_indices = parse_result
            
            run_length = len(actions_np)
            run_lengths.append(run_length)
            stage_boundaries.append((cumulative_length, cumulative_length + run_length))
            
            # Store correction info (pad with zeros if None for normal datasets)
            if correction_flags is None:
                correction_flags = np.zeros(run_length, dtype=np.float32)
            if correction_command_indices is None:
                correction_command_indices = np.zeros(run_length, dtype=np.int64) - 1
            
            all_correction_flags.append(correction_flags)
            all_correction_command_indices.append(correction_command_indices)
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
                # target_ts = curr_ts + prediction_offset should be in [stage_start, stage_end)
                # So curr_ts should be in [stage_start - prediction_offset, stage_end - prediction_offset)
                non_cross_min_curr = max(min_start, stage_start - self.prediction_offset)
                non_cross_max_curr = min(stage_end - self.prediction_offset - 1, max_start)
                
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
        
        # Get correction flag and command index for curr_ts (current frame, not target_ts)
        # Correction is an immediate feedback based on current state, not a future prediction
        # Find which run the curr_ts belongs to
        curr_run_idx = len(selected_runs) - 1  # Default to last run
        curr_local_ts = curr_ts
        for i, (start_bound, end_bound) in enumerate(stage_boundaries):
            if start_bound <= curr_ts < end_bound:
                curr_run_idx = i
                curr_local_ts = curr_ts - start_bound
                break
        
        # Get correction flag and command index for current frame
        if curr_run_idx < len(all_correction_flags) and curr_local_ts < len(all_correction_flags[curr_run_idx]):
            correction_flag = all_correction_flags[curr_run_idx][curr_local_ts]
            correction_command_idx = all_correction_command_indices[curr_run_idx][curr_local_ts]
        else:
            correction_flag = 0.0
            correction_command_idx = -1
        
        return image_sequence, command_embedding, command_gt, correction_flag, correction_command_idx


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
    normal_root_dir: str | None = None,  # Root directory for normal datasets (expert demonstrations)
    dagger_root_dir: str | None = None,  # Root directory for dagger datasets (agent trajectories with corrections)
    normal_val_root: str | None = None,  # Root directory for normal validation datasets
    dagger_val_root: str | None = None,  # Root directory for dagger validation datasets
    dagger_mix_ratio: float = 0.5,  # Ratio of dagger samples in the mixed dataset (0.0 = all normal, 1.0 = all dagger)
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
    # Support DAgger mode: if normal_root_dir and dagger_root_dir are provided, use them
    if normal_root_dir and dagger_root_dir:
        # DAgger mode: combine normal (expert) and dagger (agent with corrections) datasets
        normal_train_root = os.path.join(normal_root_dir, train_subdir) if train_subdir else normal_root_dir
        dagger_train_root = os.path.join(dagger_root_dir, train_subdir) if train_subdir else dagger_root_dir
        
        if not os.path.exists(normal_train_root):
            raise ValueError(f"Normal training directory not found: {normal_train_root}")
        if not os.path.exists(dagger_train_root):
            raise ValueError(f"Dagger training directory not found: {dagger_train_root}")
        
        print(f"DAgger mode: Separating expert and agent sequences, mixing at batch level")
        print(f"  Normal dataset: {normal_train_root}")
        print(f"  Dagger dataset: {dagger_train_root}")
        print(f"  Dagger mix ratio: {dagger_mix_ratio:.2f} (ratio of dagger samples in each batch)")
        
        # Create separate datasets for expert and agent sequences
        # This maintains sequence integrity: expert sequences show complete correct execution,
        # agent sequences show execution with corrections
        normal_dataset = CompositeSequenceDataset(
            normal_train_root,
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
        
        dagger_dataset = CompositeSequenceDataset(
            dagger_train_root,
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
        
        # Combine datasets for batch-level mixing
        train_dataset = ConcatDataset([normal_dataset, dagger_dataset])
        
        normal_size = len(normal_dataset)
        dagger_size = len(dagger_dataset)
        total_size = len(train_dataset)
        
        print(f"Training dataset: {normal_size} expert + {dagger_size} agent = {total_size} total samples")
        print(f"  Batch-level mixing: {dagger_mix_ratio:.2f} ratio of agent samples per batch")
        
        # Store sizes for later use in creating WeightedRandomSampler
        train_dataset.normal_size = normal_size
        train_dataset.dagger_size = dagger_size
        train_dataset.dagger_mix_ratio = dagger_mix_ratio
        
        # Handle validation directories for DAgger mode
        if val_root:
            val_root_dir = val_root
        else:
            # Use provided normal_val_root and dagger_val_root if available, otherwise try to find from subdirs
            if normal_val_root is None:
                normal_val_root = os.path.join(normal_root_dir, val_subdir) if val_subdir else None
            if dagger_val_root is None:
                dagger_val_root = os.path.join(dagger_root_dir, val_subdir) if val_subdir else None
            val_root_dir = None  # Will be handled separately
    else:
        # Original mode: use single root_dir
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
    if normal_root_dir and dagger_root_dir:
        # DAgger mode: combine normal and dagger validation datasets
        # Use provided normal_val_root and dagger_val_root if available, otherwise try to find from subdirs
        if normal_val_root is None:
            normal_val_root = os.path.join(normal_root_dir, val_subdir) if val_subdir else None
        if dagger_val_root is None:
            dagger_val_root = os.path.join(dagger_root_dir, val_subdir) if val_subdir else None
        
        val_datasets = []
        if val_root:
            # Use provided validation root (should contain both normal and dagger)
            if os.path.exists(val_root):
                val_dataset = CompositeSequenceDataset(
                    val_root,  # Primary root
                    camera_names,
                    history_len=history_len,
                    prediction_offset=prediction_offset,
                    history_skip_frame=history_skip_frame,
                    random_crop=random_crop,
                    stage_embeddings=stage_embeddings,
                    stage_texts=stage_texts,
                    use_augmentation=False,
                    training=False,
                    image_size=image_size,
                    sampling_strategy='sequential',
                    max_combinations=None,
                    n_repeats=None,
                    sync_sequence_augmentation=False,
                    use_weak_traversal=use_weak_traversal,
                    samples_non_cross_per_stage=samples_non_cross_per_stage,
                    samples_cross_per_stage=samples_cross_per_stage,
                    normal_root_dir=normal_val_root,  # Optional: additional normal validation data
                    dagger_root_dir=dagger_val_root,   # Optional: additional dagger validation data
                )
                print(f"Validation dataset: {len(val_dataset)} samples from {val_root}")
        else:
            # Create a single validation dataset that mixes runs from both normal and dagger validation roots
            # Use normal_val_root as primary, dagger_val_root as additional source
            if normal_val_root and os.path.exists(normal_val_root):
                val_dataset = CompositeSequenceDataset(
                    normal_val_root,  # Primary root
                    camera_names,
                    history_len=history_len,
                    prediction_offset=prediction_offset,
                    history_skip_frame=history_skip_frame,
                    random_crop=random_crop,
                    stage_embeddings=stage_embeddings,
                    stage_texts=stage_texts,
                    use_augmentation=False,
                    training=False,
                    image_size=image_size,
                    sampling_strategy='sequential',  # Sequential for deterministic validation
                    max_combinations=None,
                    n_repeats=None,
                    sync_sequence_augmentation=False,
                    use_weak_traversal=use_weak_traversal,
                    samples_non_cross_per_stage=samples_non_cross_per_stage,
                    samples_cross_per_stage=samples_cross_per_stage,
                    normal_root_dir=normal_val_root,  # Normal validation data source
                    dagger_root_dir=dagger_val_root,   # Dagger validation data source (will be mixed in)
                )
                print(f"Validation dataset: {len(val_dataset)} samples (mixed from normal and dagger validation roots)")
                if normal_val_root:
                    print(f"  Normal validation root: {normal_val_root}")
                if dagger_val_root:
                    print(f"  Dagger validation root: {dagger_val_root}")
    elif val_root_dir and os.path.exists(val_root_dir):
        # Original mode: single validation directory
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
    if normal_root_dir and dagger_root_dir:
        # DAgger mode: use WeightedRandomSampler to control batch-level mixing ratio
        normal_size = getattr(train_dataset, 'normal_size', 0)
        dagger_size = getattr(train_dataset, 'dagger_size', 0)
        dagger_mix_ratio = getattr(train_dataset, 'dagger_mix_ratio', 0.5)
        
        if normal_size > 0 and dagger_size > 0:
            # Calculate weights for WeightedRandomSampler
            # We want dagger_mix_ratio of samples to be from dagger dataset
            # So: dagger_weight / (normal_weight + dagger_weight) = dagger_mix_ratio
            # Let normal_weight = 1.0 - dagger_mix_ratio, dagger_weight = dagger_mix_ratio
            # This ensures: dagger_weight / (normal_weight + dagger_weight) = dagger_mix_ratio / 1.0 = dagger_mix_ratio
            normal_weight = 1.0 - dagger_mix_ratio
            dagger_weight = dagger_mix_ratio
            
            # Create weights: first normal_size samples get normal_weight, next dagger_size samples get dagger_weight
            weights = [normal_weight] * normal_size + [dagger_weight] * dagger_size
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),  # Sample with replacement to maintain ratio
                replacement=True
            )
            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size_train,
                sampler=sampler,  # Use WeightedRandomSampler instead of shuffle
                pin_memory=True,
                num_workers=8,
                prefetch_factor=8,
                persistent_workers=True,
            )
            print(f"  Using WeightedRandomSampler: normal_weight={normal_weight:.3f}, dagger_weight={dagger_weight:.3f}")
            print(f"  Expected ratio: {dagger_mix_ratio:.2f} dagger samples per batch")
        else:
            # Fallback: simple shuffle if sizes are invalid
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size_train,
                shuffle=True,
                pin_memory=True,
                num_workers=8,
                prefetch_factor=8,
                persistent_workers=True,
            )
        
        # Create validation dataloader if available
        if val_dataset is not None:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=False,  # No shuffle for validation
                pin_memory=True,
                num_workers=8,
                prefetch_factor=16,
                persistent_workers=True,
            )
        else:
            val_dataloader = None
    elif dagger_ratio is not None:
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

