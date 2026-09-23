import argparse
import json
from pathlib import Path
from typing import List


def parse_args():
    p = argparse.ArgumentParser(description="Validate that training artifacts exist and look healthy.")
    p.add_argument("--work-dir", type=str, required=True, help="Training work directory to inspect.")
    p.add_argument("--min-checkpoints", type=int, default=2, help="Minimum number of checkpoints expected.")
    p.add_argument("--print-json", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    if not work_dir.exists():
        raise RuntimeError(f"Work dir not found: {work_dir}")

    ckpts = sorted(work_dir.glob("iter_*.pth"))
    logs = sorted(work_dir.glob("*.log"))
    has_best = (work_dir / "best.pth").exists()
    has_last = (work_dir / "last_checkpoint").exists()

    report = {
        "work_dir": str(work_dir),
        "num_iter_checkpoints": len(ckpts),
        "num_logs": len(logs),
        "has_best": has_best,
        "has_last_checkpoint_file": has_last,
        "latest_iter_checkpoint": str(ckpts[-1]) if ckpts else None,
        "latest_log": str(logs[-1]) if logs else None,
    }

    if len(ckpts) < args.min_checkpoints:
        raise RuntimeError(
            f"Expected at least {args.min_checkpoints} checkpoints in {work_dir}, found {len(ckpts)}."
        )
    if not logs:
        raise RuntimeError(f"No log files found in {work_dir}.")

    if args.print_json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
