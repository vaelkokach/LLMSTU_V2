#!/usr/bin/env python
"""Full run, SHARD BY SHARD (steps 1-5 per shard, streamed to the Hub).

For each shard: download -> crop -> dedup -> caption -> upload crops+labels -> mark
done -> free disk. The caption model is loaded ONCE and reused across shards.
Resumable: shards already marked done on the Hub are skipped, so a Colab disconnect
just resumes with the remaining shards (nothing re-downloaded, nothing re-captioned).

  python scripts/12_run_shards.py --repo CHANGE_ME/your-crops-dataset
  python scripts/12_run_shards.py --repo CHANGE_ME/your-crops-dataset --shards shard_003,shard_004

After ALL shards are done, run steps 6-8:
  python scripts/14_golden_from_hf.py --repo CHANGE_ME/your-crops-dataset --n 1500   (build golden)
  # ...hand-label, then:
  python scripts/07_eval_golden.py --golden labeling/golden.json
  python scripts/13_finalize.py --repo CHANGE_ME/your-crops-dataset --golden labeling/golden.json
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmstu.config import load
from llmstu import caption as caption_mod, shardrun, dataset_io as io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--repo", default=None, help="crops dataset repo (default: config data.crops_repo)")
    ap.add_argument("--shards", default=None, help="comma list; default = all shard_000..NN")
    ap.add_argument("--num-shards", type=int, default=21, help="how many shards exist (for default range)")
    ap.add_argument("--token", default=None)
    args = ap.parse_args()

    cfg = load(args.config)
    repo = args.repo or cfg.data.crops_repo
    token = args.token or os.environ.get("HF_TOKEN")
    shards = (args.shards.split(",") if args.shards
              else [f"shard_{i:03d}" for i in range(args.num_shards)])

    # figure out what's left BEFORE loading the model (skip loading if nothing to do)
    todo = [s for s in shards if not io.shard_done(repo, s, token)]
    print(f"[run] {len(todo)}/{len(shards)} shards remaining: {todo or 'none'}")
    if not todo:
        print("[run] all shards already processed. Proceed to 14_golden_from_hf.py.")
        return

    cap = caption_mod.load_captioner(cfg.caption)   # load the model ONCE
    for shard in shards:
        shardrun.process_one(cfg, shard, repo, token, cap=cap)
    print("[run] all shards processed. Next: scripts/14_golden_from_hf.py")


if __name__ == "__main__":
    main()
