#!/usr/bin/env python
"""Pseudo-label crops with Qwen3-VL.

  python scripts/02_caption.py --manifest ./work/crops_manifest.jsonl \
      --crops ./work/crops --out ./work/pseudo_labels.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import caption as caption_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--manifest", default="./work/crops_manifest.jsonl")
    ap.add_argument("--crops", default="./work/crops")
    ap.add_argument("--out", default="./work/pseudo_labels.jsonl")
    ap.add_argument("--model", default=None, help="override caption.model_id")
    ap.add_argument("--backend", default=None, choices=[None, "transformers", "vllm"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    if args.model:
        cfg.caption.model_id = args.model
    if args.backend:
        cfg.caption.backend = args.backend
    caption_mod.run(Path(args.manifest), Path(args.crops), Path(args.out),
                    cfg.caption, limit=args.limit)


if __name__ == "__main__":
    main()
