#!/usr/bin/env python
"""Try several CROP configs on a few frames and render a contact-sheet HTML.

  python scripts/05_tune_crops.py --frames ./work/frames --n-frames 8

Open work/tune/crops_report.html to eyeball tightness / coverage / drops, then copy
the winning variant's values into config.yaml (crop:).
"""
import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import crop as crop_mod
from llmstu import report
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--experiments", default="experiments.yaml")
    ap.add_argument("--frames", required=True, help="dir of sample frames")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--out", default="./work/tune")
    args = ap.parse_args()

    cfg = load(args.config)
    variants = yaml.safe_load(open(args.experiments))["crop_variants"]
    frames = sorted(Path(args.frames).glob("**/*.jpg"))[: args.n_frames]
    assert frames, f"no frames under {args.frames}"
    out = Path(args.out); (out / "crops").mkdir(parents=True, exist_ok=True)

    sections = []
    for v in variants:
        c = replace(cfg.crop, **{k: v[k] for k in v if k != "name"})
        det = crop_mod._load_detector(c)
        imgs, n_crop = [], 0
        vdir = out / "crops" / v["name"]; vdir.mkdir(parents=True, exist_ok=True)
        for fr in frames:
            im = Image.open(fr).convert("RGB")
            for j, (crop, meta) in enumerate(crop_mod.crop_frame(im, det, c)):
                p = vdir / f"{fr.stem}__p{j:02d}.jpg"
                crop.save(p, quality=90); imgs.append(p); n_crop += 1
        params = f"mode={c.crop_mode} head_frac={c.head_frac} margin={c.margin_frac} det={c.detector_model} conf={c.conf}"
        stats = f"{n_crop} crops from {len(frames)} frames (avg {n_crop/len(frames):.1f}/frame)"
        sections.append({"name": v["name"], "params": params, "stats": stats, "images": imgs})
        print(f"[tune-crops] {v['name']}: {stats}")

    rp = report.contact_sheet(sections, out / "crops_report.html")
    print(f"[tune-crops] report -> {rp}")


if __name__ == "__main__":
    main()
