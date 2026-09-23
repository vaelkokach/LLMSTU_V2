#!/usr/bin/env python
"""Step 9, LOCAL: build the HF-ready metadata + parquet from extracted-on-disk shards.

Writes crops/metadata.jsonl (HF imagefolder: each crop image paired with its labels)
next to your local crops, plus pseudo_labels.parquet and (optionally) copies the
golden set. Then you can either load it locally with
`datasets.load_dataset("imagefolder", data_dir=<crops-root>)` or upload it to HF with
--repo (runs from your PC, off GPU).

  python scripts/20_finalize_local.py --labels-dir ./data/labels --crops-root ./data/crops \
      --golden labeling/golden.json
  # optional, to publish:  --repo CHANGE_ME/your-crops-dataset
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out", default="./work/final",
                    help="where metadata.jsonl + parquet + golden are written")
    ap.add_argument("--golden", default="labeling/golden.json")
    ap.add_argument("--repo", default=None, help="if set, upload --out to this HF repo")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    load(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = io.load_local_labels(args.labels_dir)
    print(f"[finalize] {len(rows)} labels loaded")

    # 1. metadata.jsonl (file_name + label fields) — HF imagefolder convention
    keep = list(schema.FIELDS) + [schema.MODEL_CONFIDENCE_FIELD, "src_frame"]
    n = 0
    with (out / "metadata.jsonl").open("w") as mf:
        for r in rows:
            fn = r.get("file_name")
            if not fn:
                continue
            rec = {"file_name": fn}
            for k in keep:
                if k in r:
                    rec[k] = r[k]
            mf.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[finalize] wrote {n} metadata rows -> {out/'metadata.jsonl'}")

    # 2. full parquet + golden copy
    tmp = out / "pseudo_labels_all.jsonl"
    tmp.write_text("".join(json.dumps({k: v for k, v in r.items() if k != '_crops_base'})
                           + "\n" for r in rows))
    io.build_parquet(tmp, out / "pseudo_labels.parquet")
    if Path(args.golden).exists():
        shutil.copy(args.golden, out / "golden_eval.json")

    # 3. optional upload to HF (from your PC, off GPU)
    if args.repo:
        token = args.token or os.environ.get("HF_TOKEN")
        io.ensure_repo(args.repo, token=token)
        io.upload_dir(out, args.repo, token=token)
        print(f"[finalize] uploaded -> https://huggingface.co/datasets/{args.repo}")
    else:
        print(f"[finalize] wrote {out}/  (metadata.jsonl, pseudo_labels.parquet, golden_eval.json)")


if __name__ == "__main__":
    main()
