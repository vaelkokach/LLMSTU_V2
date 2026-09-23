"""Launch a set of independent single-GPU training runs across the local GPUs.

Independent processes rather than DDP: the models are small, this is embarrassingly
parallel across (feature_config, seed, model), and it removes every cross-rank
reduction that this project has previously got wrong. Each run gets one GPU;
runs are dispatched to whichever GPU frees up first.

    python -m attention.thesis_eval.launch_sweep --sweep ladder --out-root work_dirs/thesis/ladder
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List

SEEDS = (42, 43, 44)
LADDER = ("552_base", "556_hp", "563_expr", "563_dyn", "570_full")


def sweep_jobs(name: str) -> List[Dict]:
    jobs: List[Dict] = []
    if name == "ladder":
        for cfg, seed in itertools.product(LADDER, SEEDS):
            jobs.append({"experiment_id": f"transformer_{cfg}_s{seed}",
                         "model": "transformer", "feature_config": cfg, "seed": seed})
    elif name == "arch":
        # Architecture comparison holds the feature block fixed at the rung the
        # ladder selects; both boundary-aware baselines see identical inputs.
        for model, seed in itertools.product(("mstcn", "asrf"), SEEDS):
            jobs.append({"experiment_id": f"{model}_556_hp_s{seed}",
                         "model": model, "feature_config": "556_hp", "seed": seed})
        for model, seed in itertools.product(("mstcn", "asrf"), SEEDS):
            jobs.append({"experiment_id": f"{model}_570_full_s{seed}",
                         "model": model, "feature_config": "570_full", "seed": seed})
    elif name == "posefix":
        # Same 556-dim rung on sequences whose pose block was recomputed from
        # the full frame + bbox_person, so training matches deployment
        # [internal notes, not included]. 552_base is unaffected — it contains no pose
        # columns — so the existing 552 runs remain the valid control.
        for model, seed in itertools.product(("transformer", "mstcn", "asrf"), SEEDS):
            jobs.append({"experiment_id": f"{model}_556_hp_bp_s{seed}",
                         "model": model, "feature_config": "556_hp", "seed": seed})
    elif name == "facefound":
        # Can a plain face DETECTOR replace the FaceLandmarker mesh?
        # 553_facefound selects base + the flag only, so this isolates the
        # question from the metric angles the detector cannot provide.
        for model, seed in itertools.product(("transformer", "mstcn"), SEEDS):
            jobs.append({"experiment_id": f"{model}_553_ff_s{seed}",
                         "model": model, "feature_config": "553_facefound", "seed": seed})
    elif name == "headpose":
        # Decomposes the one feature block that demonstrably works, to decide
        # whether a stronger head-pose estimator could add anything.
        for cfg, seed in itertools.product(("553_facefound", "555_angles"), SEEDS):
            jobs.append({"experiment_id": f"transformer_{cfg}_s{seed}",
                         "model": "transformer", "feature_config": cfg, "seed": seed})
    else:
        raise KeyError(name)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    choices=["ladder", "arch", "headpose", "posefix", "facefound"])
    ap.add_argument("--out-root", required=True)
    # Hard project constraint: at most 4 GPUs may be occupied at once, even
    # though the host exposes 8. Do not widen this default.
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sequence-root",
                    default="../grounding_data/llmstu_sequences_full")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    jobs = sweep_jobs(args.sweep)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for j in jobs:
        d = out_root / j["experiment_id"]
        if args.skip_existing and (d / "run_record.json").exists():
            print(f"skip (done): {j['experiment_id']}")
            continue
        pending.append((j, d))

    running: List[tuple] = []
    free = list(gpus)
    log_dir = out_root / "logs"
    log_dir.mkdir(exist_ok=True)
    t0 = time.time()
    while pending or running:
        while pending and free:
            j, d = pending.pop(0)
            g = free.pop(0)
            cmd = ["python", "-m", "attention.thesis_eval.train",
                   "--experiment-id", j["experiment_id"], "--model", j["model"],
                   "--feature-config", j["feature_config"], "--seed", str(j["seed"]),
                   "--epochs", str(args.epochs), "--output-dir", str(d),
                   "--device", f"cuda:{g}", "--threads", str(args.threads),
                   "--sequence-root", args.sequence_root]
            log = open(log_dir / f"{j['experiment_id']}.log", "w")
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            running.append((p, g, j["experiment_id"], log))
            print(f"launch {j['experiment_id']} on cuda:{g}", flush=True)
        time.sleep(5)
        for item in list(running):
            p, g, eid, log = item
            if p.poll() is not None:
                log.close()
                running.remove(item)
                free.append(g)
                status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                print(f"[{time.time()-t0:6.0f}s] {eid}: {status}", flush=True)

    records = {}
    for j in jobs:
        f = out_root / j["experiment_id"] / "run_record.json"
        if f.exists():
            r = json.loads(f.read_text())
            records[j["experiment_id"]] = {
                "model": r["spec"]["model"], "feature_config": r["spec"]["feature_config"],
                "seed": r["spec"]["seed"], "selected_epoch": r["selected_epoch"],
                "selected_val_macro_f1": r["selected_val_macro_f1"],
                "wall_clock_s": r["wall_clock_s"]}
    (out_root / "sweep_summary.json").write_text(json.dumps(records, indent=2))
    print(f"\n{len(records)}/{len(jobs)} runs complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
