import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path


def copy_required(src: Path, dst: Path, what: str, optional: bool = False) -> bool:
    """Copy src -> dst; missing sources are a hard error unless optional."""
    if not src.exists():
        if optional:
            print(f"[skip] optional {what}: {src} does not exist")
            return False
        sys.exit(f"ERROR: required {what} not found: {src}\n"
                 "Refusing to write a bundle with missing/placeholder artifacts.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Collect thesis artifacts into one reproducible bundle.")
    p.add_argument("--bundle-dir", type=str, default="work_dirs/thesis_bundle")
    p.add_argument("--e0-ckpt", type=str, default="work_dirs/grounding_dino_swin_t/iter_15000.pth")
    p.add_argument("--e1-best", type=str, required=True)
    p.add_argument("--e2-best", type=str, default=None,
                   help="Omit if E2 was never trained; the bundle then simply has no E2 entry.")
    p.add_argument("--temporal-best", type=str, default="work_dirs/attention_temporal/checkpoints/best.pth")
    p.add_argument("--temporal-metrics-json", type=str, default="work_dirs/attention_temporal/checkpoints/best_val_class_metrics.json")
    p.add_argument("--temporal-cm-npy", type=str, default="work_dirs/attention_temporal/checkpoints/best_val_confusion_matrix.npy")
    p.add_argument("--ablation-root", type=str, default="work_dirs/ablation_e0_e1")
    p.add_argument("--hybrid-video", type=str, default="work_dirs/attention_temporal/realtime_e1_hybrid.mp4")
    p.add_argument("--detection-metrics", type=str, nargs="*", default=[
        "work_dirs/ablation_e0_e1/e0/20260325_193738/20260325_193738.json",
        "work_dirs/ablation_e0_e1/e1/20260325_200039/20260325_200039.json",
        "work_dirs/grounding_dino_swin_t_student_only/20260328_203639/vis_data/20260328_203639.json",
    ], help="Scalar/eval JSON logs to copy into metrics/detection/.")
    return p.parse_args()


def main():
    args = parse_args()
    bundle = Path(args.bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    copy_required(Path(args.e0_ckpt), bundle / "checkpoints" / "e0_iter15000.pth", "E0 checkpoint")
    copy_required(Path(args.e1_best), bundle / "checkpoints" / "e1_best.pth", "E1 checkpoint")

    if args.e2_best is not None:
        e2 = Path(args.e2_best)
        copy_required(e2, bundle / "checkpoints" / "e2_best.pth", "E2 checkpoint")
        # Guard against re-bundling a mislabeled duplicate (a previous bundle shipped
        # e2_best.pth as a byte-copy of E0).
        if filecmp.cmp(e2, Path(args.e0_ckpt), shallow=False):
            sys.exit(f"ERROR: --e2-best {e2} is byte-identical to --e0-ckpt {args.e0_ckpt}. "
                     "That is not a trained E2 model; drop --e2-best or point it at a real checkpoint.")

    copy_required(Path(args.temporal_best), bundle / "checkpoints" / "temporal_best.pth", "temporal checkpoint")
    copy_required(Path(args.temporal_metrics_json), bundle / "metrics" / "best_val_class_metrics.json", "temporal metrics")
    copy_required(Path(args.temporal_cm_npy), bundle / "metrics" / "best_val_confusion_matrix.npy", "confusion matrix")
    copy_required(Path(args.ablation_root), bundle / "ablation", "ablation results dir")
    copy_required(Path(args.hybrid_video), bundle / "qualitative" / "realtime_hybrid.mp4", "hybrid demo video", optional=True)

    detection_copied = []
    for m in args.detection_metrics:
        src = Path(m)
        dst = bundle / "metrics" / "detection" / (src.parent.name + "_" + src.name
                                                  if src.name in ("scalars.json",) else src.name)
        if copy_required(src, dst, "detection metrics json", optional=True):
            detection_copied.append(str(dst.relative_to(bundle)))

    manifest = {
        "e0_ckpt": args.e0_ckpt,
        "e1_best": args.e1_best,
        "e2_best": args.e2_best,
        "temporal_best": args.temporal_best,
        "temporal_metrics_json": args.temporal_metrics_json,
        "temporal_cm_npy": args.temporal_cm_npy,
        "ablation_root": args.ablation_root,
        "hybrid_video": args.hybrid_video,
        "detection_metrics": detection_copied,
    }
    with (bundle / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Thesis bundle written to: {bundle}")


if __name__ == "__main__":
    main()
