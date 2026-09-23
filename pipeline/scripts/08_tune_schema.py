#!/usr/bin/env python
"""A/B different FIELD SETS (schemas) on a fixed crop set.

Where 06_tune_captions compares prompt WORDING (fixed fields), this compares the
FIELDS themselves — e.g. full vs. dropping attention_target vs. merged gaze
categories. For each variant it captions the same crops under that schema and
reports each enum field's "concrete-rate" (how often the model answered with
something other than "unknown"). Fields that are mostly "unknown" are ambiguous
under that schema and are candidates to drop or simplify.

  python scripts/08_tune_schema.py --crops ./work/tune_crops --n-crops 24

Edit schema variants in experiments.yaml (schema_variants) and the field sets
themselves in llmstu/schema.py (SCHEMAS). Confirm correctness later with the golden
set (07_eval_golden); this step measures decidability, not correctness.
"""
import argparse
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
    ap.add_argument("--crops", required=True)
    ap.add_argument("--n-crops", type=int, default=24)
    ap.add_argument("--out", default="./work/tune")
    args = ap.parse_args()

    cfg = load(args.config)
    variants = yaml.safe_load(open(args.experiments))["schema_variants"]
    crop_paths = sorted(Path(args.crops).glob("*.jpg"))[: args.n_crops]
    assert crop_paths, f"no crops under {args.crops}"
    images = [Image.open(p).convert("RGB") for p in crop_paths]

    results = []
    for v in variants:
        overrides = {k: v[k] for k in v if k not in ("name", "schema")}
        c = replace(cfg.caption, schema_name=v["schema"], **overrides)
        print(f"[tune-schema] {v['name']} (schema={v['schema']}, model={c.model_id})")
        cap = caption_mod.load_captioner(c)          # activates the schema internally
        fields = schema.SCHEMAS[v["schema"]]
        enum_fields = [f for f, s in fields.items() if s["type"] == "enum"]

        labels = []
        for s in range(0, len(images), c.batch_size):
            for res in cap.caption_batch(images[s:s + c.batch_size]):
                labels.append(res["label"])

        coverage = []
        for f in enum_fields:
            concrete = sum(1 for lab in labels
                           if str(lab.get(f, "unknown")).lower() != "unknown")
            coverage.append({"field": f, "pct": 100 * concrete / max(1, len(labels))})
        confs = [lab.get(schema.MODEL_CONFIDENCE_FIELD) for lab in labels]
        confs = [x for x in confs if isinstance(x, (int, float))]
        results.append({"name": v["name"], "schema": v["schema"],
                        "n_fields": len(fields), "coverage": coverage,
                        "mean_conf": (sum(confs) / len(confs)) if confs else 0.0})
        del cap
        try:
            import torch, gc; gc.collect(); torch.cuda.empty_cache()
        except Exception:
            pass

    rp = report.schema_report(results, Path(args.out) / "schema_report.html")
    print(f"[tune-schema] report -> {rp}")


if __name__ == "__main__":
    main()
