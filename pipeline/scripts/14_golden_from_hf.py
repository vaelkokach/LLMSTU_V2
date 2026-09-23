#!/usr/bin/env python
"""Build the golden-eval sample from crops already uploaded to the Hub (shard-wise run).

Gathers all shard labels from the crops repo, samples (stratified representative +
hard subset), downloads ONLY the sampled crop images, and writes a self-contained
labeling payload. Run this after all shards are done.

  python scripts/14_golden_from_hf.py --repo CHANGE_ME/your-crops-dataset --n 1500
"""
import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, sampling, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out", default="./labeling")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--stratify-fields", default="engagement_level,activity,gaze_direction")
    ap.add_argument("--min-per-class", type=int, default=40)
    ap.add_argument("--hard-frac", type=float, default=0.2)
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    n_total = args.n or cfg.eval.n_samples
    fields = [f.strip() for f in args.stratify_fields.split(",") if f.strip()]
    work = Path(cfg.data.workdir)

    # 1. gather all shard labels into one file (this is your full pseudo_labels set)
    all_labels = work / "pseudo_labels_all.jsonl"
    io.gather_labels(repo, all_labels, token=args.token)
    rows = [json.loads(l) for l in all_labels.open() if l.strip()]

    # 2. sample representative + hard
    picks = sampling.select(rows, n_total, fields, args.min_per_class,
                            args.hard_frac, seed=cfg.eval.seed)

    # 3. download only the sampled crops from the Hub
    sel_rows = [rows[i] for i, _ in picks]
    file_names = [r["file_name"] for r in sel_rows if "file_name" in r]
    local = io.fetch_crops(repo, file_names, work / "golden_crops", token=args.token)

    # 4. build the self-contained labeling payload (embedded images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = []
    for k, (i, group) in enumerate(picks):
        r = rows[i]
        fn = r.get("file_name")
        p = local.get(fn)
        if not p or not Path(p).exists():
            continue
        b = base64.b64encode(Path(p).read_bytes()).decode()
        data.append({"id": k, "crop_path": r.get("crop_path", fn), "group": group,
                     "pseudo": {kk: r.get(kk) for kk in schema.FIELDS},
                     "model_confidence": r.get(schema.MODEL_CONFIDENCE_FIELD),
                     "image": f"data:image/jpeg;base64,{b}"})
    (out / "data.json").write_text(json.dumps({"fields": schema.FIELDS, "items": data}, indent=1))
    nr = sum(1 for d in data if d["group"] == "representative")
    nh = sum(1 for d in data if d["group"] == "hard")
    print(f"[golden] {len(data)} items ({nr} representative + {nh} hard) -> {out/'data.json'}")
    print(f"[golden] total pseudo-labels across shards: {len(rows)}")


if __name__ == "__main__":
    main()
