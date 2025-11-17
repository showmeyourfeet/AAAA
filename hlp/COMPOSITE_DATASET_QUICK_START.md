# CompositeSequenceDataset Quick Start

## TL;DR

Use `--n_repeats N` to control how many times each run is used:

```bash
# Each run used 10 times → 1800 samples (for 180 runs/stage)
python hlp/train.py --use_composite --n_repeats 10 ...

# Each run used 50 times → 9000 samples
python hlp/train.py --use_composite --n_repeats 50 ...

# All combinations: each run used 180 times → 32400 samples
python hlp/train.py --use_composite --n_repeats 180 ...
```

## Formula

For balanced sampling:
```
max_combinations = n_repeats × min_runs_per_stage
```

**Example:** 2 stages, 180 runs each
- `n_repeats=10` → 10 × 180 = **1800 samples**
- `n_repeats=50` → 50 × 180 = **9000 samples**
- `n_repeats=180` → 180 × 180 = **32400 samples** (full combinations)

## Complete Training Command

```bash
python hlp/train.py \
  --use_composite \
  --n_repeats 10 \
  --splitted_root /path/to/training_datasets \
  --val_splitted_root /path/to/validation_datasets \
  --stage_embeddings_file /path/to/stage_embeddings.json \
  --camera_names left_frame right_frame \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 1e-5 \
  --seed 0 \
  --gpu 0 \
  --history_len 4 \
  --history_skip_frame 30 \
  --prediction_offset 15 \
  --ckpt_dir ./hlp/ckpts
```

## Key Features

### Training Set
- **Balanced sampling:** Each run used exactly `n_repeats` times (when stages have equal runs)
- **Variety:** Combinations are shuffled and varied
- **Fixed per epoch:** Combinations are pre-generated, only `current_frame` is random

### Validation Set
- **Sequential matching:** `run_001+run_001`, `run_002+run_002`, ...
- **Deterministic:** Same combinations every epoch (only `current_frame` varies)
- **No augmentation:** Clean evaluation

## Comparison with Other Modes

| Mode | Training Samples | Each Run Used | Validation | Use Case |
|------|-----------------|---------------|------------|----------|
| `--use_splitted` | 180 (one per run) | 1 time | Split from train | Quick testing |
| `--use_composite --n_repeats 10` | 1,800 | 10 times | Sequential (38) | Moderate training |
| `--use_composite --n_repeats 180` | 32,400 | 180 times | Sequential (38) | Full training |

## Monitoring

Watch for these messages:

```
[CompositeSequenceDataset] Calculated max_combinations = 10 × 180 = 1800
[CompositeSequenceDataset] Generated 1800 balanced combinations
[CompositeSequenceDataset] Combination statistics:
  Stage 0: 180 unique runs, usage range [10, 10], mean 10.0, std 0.0
  Stage 1: 180 unique runs, usage range [10, 10], mean 10.0, std 0.0
Training dataset: 1800 samples from /path/to/training_datasets

[CompositeSequenceDataset] Sequential matching:
  - Total unique run numbers: 38
  - Run numbers present in all stages: 38
[CompositeSequenceDataset] Generated 38 sequential combinations
Validation dataset: 38 samples from /path/to/validation_datasets

Train Samples: 1710  # 1800 × 0.95 (if split)
Val Samples: 38
```

## See Also

- [Full Documentation](COMPOSITE_DATASET_USAGE.md)
- [Balanced Sampling Algorithm](test_balanced_sampling.py)

