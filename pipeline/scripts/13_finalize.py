#!/usr/bin/env python
"""Finalize the shard-wise dataset: build the combined metadata + parquet and push
the golden set. Crops + per-shard labels are already on the Hub (from 12_run_shards).

  python scripts/13_finalize.py --repo CHANGE_ME/your-crops-dataset --golden labeling/golden.json

Produces in the repo:
  crops/metadata.jsonl     (HF imagefolder: every crop image paired with its labels)
  pseudo_labels.parquet    (full label table)
  golden_eval.json         (your hand-verified set)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--golden", default="labeling/golden.json")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    work = Path(cfg.data.workdir)

    # gather all shard labels
    all_labels = work / "pseudo_labels_all.jsonl"
    io.gather_labels(repo, all_labels, token=args.token)
    rows = [json.loads(l) for l in all_labels.open() if l.strip()]

    stage = work / "finalize"
    (stage / "crops").mkdir(parents=True, exist_ok=True)

    # 1. HF imagefolder metadata: file_name (relative to crops/) + label fields
    keep = list(schema.FIELDS) + [schema.MODEL_CONFIDENCE_FIELD, "src_frame"]
    with (stage / "crops" / "metadata.jsonl").open("w") as mf:
        for r in rows:
            if "file_name" not in r:
                continue
            rec = {"file_name": r["file_name"]}
            for k in keep:
                if k in r:
                    rec[k] = r[k]
            mf.write(json.dumps(rec) + "\n")

    # 2. full parquet
    io.build_parquet(all_labels, stage / "pseudo_labels.parquet")

    # 3. golden set
    if Path(args.golden).exists():
        import shutil
        shutil.copy(args.golden, stage / "golden_eval.json")

    io.upload_dir(stage, repo, token=args.token)
    print(f"[finalize] metadata.jsonl + parquet + golden pushed -> "
          f"https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
