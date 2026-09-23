#!/usr/bin/env python
"""Try several CAPTION configs (model / prompt / params) on a fixed crop set and
render a side-by-side comparison HTML with per-field agreement.

  python scripts/06_tune_captions.py --crops ./work/crops --n-crops 24

Open work/tune/captions_report.html. Pick the variant whose JSON is most correct
and consistent, then set config.yaml (caption.model_id / caption.prompt_name).

Note: each variant loads its model, so this runs them sequentially. Keep --n-crops
small (16-32) while iterating. This is compute-heavy by design (real VLM calls,
weights on the GPU — no API).
"""
import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import caption as caption_mod
from llmstu import report, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--experiments", default="experiments.yaml")
    ap.add_argument("--crops", required=True, help="dir of crop .jpgs to label")
    ap.add_argument("--n-crops", type=int, default=24)
    ap.add_argument("--out", default="./work/tune")
    args = ap.parse_args()

    cfg = load(args.config)
    variants = yaml.safe_load(open(args.experiments))["caption_variants"]
    crop_paths = sorted(Path(args.crops).glob("*.jpg"))[: args.n_crops]
    assert crop_paths, f"no crops under {args.crops}"
    images = [Image.open(p).convert("RGB") for p in crop_paths]

    # collect labels[variant][crop_idx]
    per_variant = {}
    for v in variants:
        c = replace(cfg.caption, **{k: v[k] for k in v if k != "name"})
        print(f"[tune-caps] loading {v['name']} ({c.model_id}, prompt={c.prompt_name}, 4bit={c.load_in_4bit})")
        cap = caption_mod.load_captioner(c)
        labels = []
        bs = c.batch_size
        for s in range(0, len(images), bs):
            for res in cap.caption_batch(images[s:s + bs]):
                labels.append(res["label"])
        per_variant[v["name"]] = labels
        del cap
        try:
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    vnames = [v["name"] for v in variants]
    fields = list(schema.FIELDS)
    crops = []
    for i, p in enumerate(crop_paths):
        crops.append({"image": p,
                      "labels": {vn: per_variant[vn][i] for vn in vnames}})
    rp = report.caption_comparison(crops, vnames, fields, Path(args.out) / "captions_report.html")
    print(f"[tune-caps] report -> {rp}")


if __name__ == "__main__":
    main()
