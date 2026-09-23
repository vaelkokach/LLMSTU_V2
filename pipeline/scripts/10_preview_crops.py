#!/usr/bin/env python
"""Fast visual crop tuning — crop a few frames with knob overrides, save to a folder.

No captioning, no golden set. Tweak the flags, re-run, look at the crops, repeat.
Runs as a subprocess so it always uses the latest code (no kernel reload), and the
pose model is small so it fits even if a big model is loaded elsewhere.

  python scripts/10_preview_crops.py --frames work/frames_dl/frames --n-frames 6 \
      --down-extend 2.5 --margin-frac 0.30 --min-native-px 120 \
      --blur-min-var 40 --min-face-kpts 2 --out work/preview

Then in the notebook:
  import glob; from IPython.display import Image, display
  for p in sorted(glob.glob('work/preview/*.jpg')): display(Image(p, width=220))
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from llmstu.config import load
from llmstu import crop as crop_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--out", default="work/preview")
    # knob overrides (leave unset to use config.yaml)
    ap.add_argument("--down-extend", type=float)
    ap.add_argument("--margin-frac", type=float)
    ap.add_argument("--min-native-px", type=int)
    ap.add_argument("--blur-min-var", type=float)
    ap.add_argument("--min-face-kpts", type=int)
    ap.add_argument("--kpt-conf", type=float)
    ap.add_argument("--out-size", type=int, default=0,
                    help="0 = save native crop (see true framing); else square-pad to N")
    args = ap.parse_args()

    cfg = load(args.config)
    c = cfg.crop
    for attr, val in [("down_extend", args.down_extend), ("margin_frac", args.margin_frac),
                      ("min_native_px", args.min_native_px), ("blur_min_var", args.blur_min_var),
                      ("min_face_kpts", args.min_face_kpts), ("kpt_conf", args.kpt_conf)]:
        if val is not None:
            setattr(c, attr, val)
    c.out_size = args.out_size

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    det = crop_mod._load_detector(c)
    frames = sorted(Path(args.frames).glob("**/*.jpg"))[: args.n_frames]
    assert frames, f"no frames under {args.frames}"
    stats, n = {}, 0
    for f in frames:
        img = Image.open(f).convert("RGB")
        for j, (crop, meta) in enumerate(crop_mod.crop_frame(img, det, c, stats)):
            crop.save(out / f"{f.stem}__p{j:02d}.jpg", quality=92)
            n += 1
    print(f"[preview] {n} crops from {len(frames)} frames -> {out}")
    print(f"[preview] settings: down_extend={c.down_extend} margin_frac={c.margin_frac} "
          f"min_native_px={c.min_native_px} blur_min_var={c.blur_min_var} "
          f"min_face_kpts={c.min_face_kpts}")
    drops = {k: v for k, v in stats.items() if k != "kept"}
    if drops:
        print(f"[preview] dropped -> " + ", ".join(f"{k}:{v}" for k, v in drops.items()))


if __name__ == "__main__":
    main()
