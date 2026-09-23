#!/usr/bin/env python
"""Build the golden set when the shards were uploaded to HF as ZIP files.

(Each zip holds crops/<shard>/part_XXX/*.jpg + labels/<shard>.jsonl inside.)
Downloads each shard zip once (1 file each), reads the labels straight from the zips,
samples (stratified + hard), and extracts ONLY the sampled crops for labeling.

  python scripts/18_golden_from_zips.py --repo CHANGE_ME/your-frames-dataset --n 1500 \
      --stratify-fields engagement_level,activity,gaze_direction --min-per-class 40 --hard-frac 0.2
"""
import argparse
import base64
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import schema, sampling


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

    from huggingface_hub import HfApi, hf_hub_download
    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    token = args.token or os.environ.get("HF_TOKEN")
    n_total = args.n or cfg.eval.n_samples
    fields = [f.strip() for f in args.stratify_fields.split(",") if f.strip()]

    api = HfApi(token=token)
    zips = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".zip")]
    assert zips, f"no .zip shard files found in {repo}"
    print(f"[golden] {len(zips)} shard zips found")

    # 1. download each zip once, read labels straight out of it
    rows, file_to_zip = [], {}
    for zf in zips:
        p = hf_hub_download(repo, zf, repo_type="dataset", token=token)
        with zipfile.ZipFile(p) as z:
            for m in z.namelist():
                if m.startswith("labels/") and m.endswith(".jsonl"):
                    for line in z.read(m).decode().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            rows.append(r)
                            if "file_name" in r:
                                file_to_zip[r["file_name"]] = p
    print(f"[golden] gathered {len(rows)} labels")

    # 2. sample representative + hard
    picks = sampling.select(rows, n_total, fields, args.min_per_class,
                            args.hard_frac, seed=cfg.eval.seed)

    # 3. extract ONLY the sampled crops (read bytes from their zips) and embed
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    open_zips, data = {}, []
    for k, (i, group) in enumerate(picks):
        r = rows[i]
        fn = r.get("file_name")
        zp = file_to_zip.get(fn)
        if not zp:
            continue
        z = open_zips.setdefault(zp, zipfile.ZipFile(zp))
        try:
            b = z.read("crops/" + fn)
        except KeyError:
            continue
        data.append({"id": k, "crop_path": r.get("crop_path", fn), "group": group,
                     "pseudo": {kk: r.get(kk) for kk in schema.FIELDS},
                     "model_confidence": r.get(schema.MODEL_CONFIDENCE_FIELD),
                     "image": "data:image/jpeg;base64," + base64.b64encode(b).decode()})
    (out / "data.json").write_text(json.dumps({"fields": schema.FIELDS, "items": data}, indent=1))
    nr = sum(1 for d in data if d["group"] == "representative")
    nh = sum(1 for d in data if d["group"] == "hard")
    print(f"[golden] {len(data)} items ({nr} representative + {nh} hard) -> {out/'data.json'}")


if __name__ == "__main__":
    main()
