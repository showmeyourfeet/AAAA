import argparse
import os
from bisect import bisect_left

from dataset import AnnotationStageDataset


def resolve_run_index(dataset: AnnotationStageDataset, run_id: str | int) -> int:
    """Return the index for the desired run inside dataset.runs."""
    normalized = dataset._normalize_run_name(run_id)  # type: ignore[attr-defined]
    for idx, rec in enumerate(dataset.runs):
        if os.path.basename(rec["run_dir"]) == normalized:
            return idx
    raise ValueError(f"Run {run_id} (resolved to {normalized}) not found in dataset.")


def locate_start_index(timestamps: list[int], target_timestamp: int) -> int:
    """Find the frame index whose timestamp is >= target_timestamp."""
    idx = bisect_left(timestamps, target_timestamp)
    if idx >= len(timestamps):
        raise ValueError(
            f"start_ts {target_timestamp} is beyond the last timestamp ({timestamps[-1]})"
        )
    return idx


def main():
    parser = argparse.ArgumentParser(description="Test AnnotationStageDataset sampling.")
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--run_id", required=True, type=str, help="e.g. run_002 or 2")
    parser.add_argument("--start_ts", required=True, type=int, help="Timestamp in ms")
    parser.add_argument("--history_len", required=True, type=int)
    parser.add_argument("--skip_frame", required=True, type=int)
    parser.add_argument("--offset", required=True, type=int, help="Prediction offset")
    parser.add_argument("--stage_embeddings_file", type=str, default=None)
    parser.add_argument("--stage_texts_file", type=str, default=None)
    parser.add_argument("--drop_initial_frames", type=int, default=1)
    args = parser.parse_args()

    dataset = AnnotationStageDataset(
        dataset_dir=args.dataset_dir,
        run_ids=[args.run_id],
        history_len=args.history_len,
        prediction_offset=args.offset,
        history_skip_frame=args.skip_frame,
        use_augmentation=False,
        training=False,
        stage_embeddings_file=args.stage_embeddings_file,
        stage_texts_file=args.stage_texts_file,
        drop_initial_frames=args.drop_initial_frames,
    )

    run_idx = resolve_run_index(dataset, args.run_id)
    run_rec = dataset.runs[run_idx]
    timestamps = run_rec["timestamps"]

    start_index = locate_start_index(timestamps, args.start_ts)
    required_history = dataset.required_history
    if start_index < required_history:
        raise ValueError(
            f"start_ts requires at least {required_history} frames of history; "
            f"got start_index {start_index}"
        )

    target_index = start_index + dataset.prediction_offset
    if target_index >= len(run_rec["frames"]):
        raise ValueError(
            f"Target frame index {target_index} exceeds total frames ({len(run_rec['frames'])})."
        )

    history_indices = list(
        range(start_index - required_history, start_index + 1, dataset.history_skip_frame)
    )
    history_frames = [
        os.path.basename(run_rec["frames"][idx][1]) for idx in history_indices
    ]
    history_timestamps = [run_rec["frames"][idx][0] for idx in history_indices]

    target_stage = dataset._stage_for_index(run_rec, target_index)
    if target_stage is None:
        raise RuntimeError("Could not locate stage for the requested target frame.")

    target_frame_name = os.path.basename(run_rec["frames"][target_index][1])
    stage_text_from_file = dataset.stage_texts.get(target_stage["stage_id"])
    stage_embedding = dataset._embedding_for_stage(target_stage["stage_id"])

    print("=== AnnotationStageDataset Test ===")
    print(f"Run directory: {run_rec['run_dir']}")
    print(f"History indices: {history_indices}")
    print("History frames (timestamp -> filename):")
    for ts, name in zip(history_timestamps, history_frames):
        print(f"  {ts} ms -> {name}")
    print(f"\nTarget frame index: {target_index}")
    print(f"Target frame filename: {target_frame_name}")
    print(
        f"Target stage: id={target_stage['stage_id']} "
        f"name={target_stage['stage_name']} "
        f"range=({target_stage['start_ms']}, {target_stage['end_ms']})"
    )
    if stage_text_from_file:
        print(f"Stage text (from stage_texts_file): {stage_text_from_file}")
    print(
        f"Stage embedding dim={stage_embedding.shape[0]} "
        f"sample={stage_embedding[:5].tolist() if stage_embedding.numel() else []}"
    )


if __name__ == "__main__":
    main()

