from pathlib import Path

from attention.sequence_builder import build_sequences_llmstu, parse_args


if __name__ == "__main__":
    args = parse_args()
    if args.format != "llmstu":
        raise SystemExit("Legacy path is deprecated for building; use --format llmstu.")
    build_sequences_llmstu(
        label_dir=Path(args.labels),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        frame_to_video_path=Path(args.frame_to_video),
        min_track_len=args.min_track_len,
        max_track_len=args.max_track_len,
        val_fraction=args.val_fraction,
        seed=args.seed,
        allow_filename_fallback=args.allow_filename_video_fallback,
        allow_clip_fallback=args.allow_clip_fallback,
    )
