#!/usr/bin/env python
"""Build the golden-eval sample + labeling payload.

Two-part sample so it's both representative AND stresses the model:
  * REPRESENTATIVE — stratified across several fields, with a per-class floor so
    rare categories (e.g. raising_hand) get enough examples to measure. This is
    your unbiased headline-accuracy set.
  * HARD — extra crops drawn from the LOWEST model_confidence, where errors
    concentrate. Kept as a separate group so it doesn't bias the headline number.

  python scripts/03_sample_eval.py --labels work/pseudo_labels.jsonl --crops work/crops \
      --n 1500 --stratify-fields engagement_level,activity,gaze_direction \
      --min-per-class 40 --hard-frac 0.2 --embed
"""
import argparse
import base64
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, sampling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels", default="./work/pseudo_labels.jsonl")
    ap.add_argument("--crops", default="./work/crops")
    ap.add_argument("--out", default="./labeling")
    ap.add_argument("--n", type=int, default=None, help="total target sample size")
    ap.add_argument("--stratify-fields", default="engagement_level,activity,gaze_direction",
                    help="comma-separated fields to balance the representative set over")
    ap.add_argument("--min-per-class", type=int, default=40,
                    help="floor of examples per category value per stratify field")
    ap.add_argument("--hard-frac", type=float, default=0.2,
                    help="fraction of --n drawn from lowest-confidence crops (hard set)")
    ap.add_argument("--embed", action="store_true",
                    help="base64-embed images into data.json (single-file, no server)")
    args = ap.parse_args()

    cfg = load(args.config)
    n_total = args.n or cfg.eval.n_samples
    fields = [f.strip() for f in args.stratify_fields.split(",") if f.strip()]

    rows = [json.loads(l) for l in Path(args.labels).open() if l.strip()]
    picks = sampling.select(rows, n_total, fields, args.min_per_class,
                            args.hard_frac, seed=cfg.eval.seed)

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    crops = Path(args.crops)

    data = []
    for k, (i, group) in enumerate(picks):
        r = rows[i]
        src = crops / Path(r["crop_path"]).name
        if not src.exists():
            src = crops.parent / r["crop_path"]
        if not src.exists():
            continue
        item = {"id": k, "crop_path": r["crop_path"], "group": group,
                "pseudo": {kk: r.get(kk) for kk in schema.FIELDS},
                "model_confidence": r.get(schema.MODEL_CONFIDENCE_FIELD)}
        if args.embed:
            b = base64.b64encode(src.read_bytes()).decode()
            item["image"] = f"data:image/jpeg;base64,{b}"
        else:
            shutil.copy(src, img_dir / src.name)
            item["image"] = f"images/{src.name}"
        data.append(item)

    (out / "data.json").write_text(json.dumps({
        "fields": schema.FIELDS, "items": data}, indent=1))
    n_repr_out = sum(1 for d in data if d["group"] == "representative")
    n_hard_out = sum(1 for d in data if d["group"] == "hard")
    print(f"[eval] {len(data)} items ({n_repr_out} representative + {n_hard_out} hard) "
          f"-> {out/'data.json'}  (open {out/'index.html'} to label)")
    print(f"[eval] stratified over {fields} with floor {args.min_per_class}/class")


if __name__ == "__main__":
    main()
