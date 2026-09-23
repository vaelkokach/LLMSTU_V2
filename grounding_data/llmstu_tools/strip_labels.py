#!/usr/bin/env python3
"""Merge the 21 LLMSTU label shards into one slim JSONL.

Drops `raw_caption` (byte-for-byte redundant re-serialisation of the other
fields, ~45% of the file size) and `crop_path` (legacy flat path that does not
resolve on disk; `file_name` is the real relative path under crops/).

Output: <llmstu_root>/labels_slim.jsonl
"""
import argparse
import glob
import json
import os

DROP_KEYS = ("raw_caption", "crop_path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llmstu-root",
                    default="/home/jovyan/Computer_vision/grounding_data/LLMSTU")
    ap.add_argument("--out", default=None,
                    help="default: <llmstu-root>/labels_slim.jsonl")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.llmstu_root, "labels_slim.jsonl")
    shards = sorted(glob.glob(os.path.join(args.llmstu_root, "labels", "shard_*.jsonl")))
    assert shards, f"no label shards under {args.llmstu_root}/labels"

    n = 0
    seen = set()
    with open(out_path, "w") as out:
        for shard in shards:
            with open(shard) as fh:
                for line in fh:
                    rec = json.loads(line)
                    for k in DROP_KEYS:
                        rec.pop(k, None)
                    fn = rec["file_name"]
                    assert fn not in seen, f"duplicate file_name {fn}"
                    seen.add(fn)
                    out.write(json.dumps(rec) + "\n")
                    n += 1
    print(f"wrote {n} records -> {out_path}")


if __name__ == "__main__":
    main()
