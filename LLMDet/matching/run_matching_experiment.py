"""Compare ordinal / ordinal_lr / hungarian / sinkhorn on exact LLMSTU GT.

CPU-only by default. Example:

    python -m matching.run_matching_experiment --frames 500 \
        --unit-order lr --dump work_dirs/matching/assignments.jsonl

Unit-order semantics (the crux of the experiment): the detector emits boxes in
confidence order (verified in the audit — there is no spatial sort in the
pipeline), while the VLM numbers "Student 1..N" in an unknown order, most
plausibly left-to-right reading order. --unit-order picks the simulated VLM
order; each choice makes the corresponding ordinal baseline an upper bound
(lr -> ordinal_lr is perfect by construction, conf -> ordinal is perfect), so
the content-based matchers' order-invariant accuracy is the meaningful
comparison.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matching.compatibility import (ClipScorer, CompatibilityWeights,
                                    assignment_components, build_cost_matrix,
                                    spatial_agreement)
from matching.evaluate_matching import (LLMSTU_ROOT, MatchingEvaluator,
                                        format_results_table, load_gt)
from matching.student_unit_matcher import (StudentUnit, match_hungarian,
                                           match_ordinal, match_ordinal_lr,
                                           match_sinkhorn)

STRATEGIES = ("ordinal", "ordinal_lr", "hungarian", "sinkhorn")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path,
                    default=Path("/home/jovyan/Computer_vision/grounding_data/"
                                 "llmstu_tools/outputs/correspondence_gt.jsonl"),
                    help="W1 ground-truth file; falls back to a stand-in "
                         "built from --standin-shard when missing")
    ap.add_argument("--standin-shard", type=Path,
                    default=LLMSTU_ROOT / "labels" / "shard_000.jsonl")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--unit-order", choices=("lr", "conf"), default="lr")
    ap.add_argument("--spatial-weight", type=float, default=0.0,
                    help="weight of the reading-order prior in the cost "
                         "(0 = pure content matching)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eps", type=float, default=0.1, help="sinkhorn epsilon")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write per-assignment components + correctness "
                         "(hungarian strategy) for calibration.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    gt = load_gt(args.gt if args.gt.exists() else None, args.standin_shard,
                 min_students=2, max_frames=None)
    rng.shuffle(gt)
    gt = gt[:args.frames]
    used_standin = not args.gt.exists()
    print(f"frames: {len(gt)}  (gt source: "
          f"{'STAND-IN from ' + str(args.standin_shard) if used_standin else args.gt})")

    scorer = ClipScorer(device=args.device)
    weights = CompatibilityWeights(clip=1.0, spatial=args.spatial_weight)
    evaluators = {s: MatchingEvaluator() for s in STRATEGIES}
    dump_records = []
    t0 = time.time()
    n_crops = 0

    for fi, frame in enumerate(gt):
        students = frame["students"]
        n = len(students)
        # presented box order = detector confidence order (the real pipeline's)
        conf_order = np.argsort([-s["det_conf"] for s in students], kind="stable")
        boxes = np.array([students[i]["bbox_person"] for i in conf_order])
        det_conf = np.array([students[i]["det_conf"] for i in conf_order])
        # simulated VLM unit order
        if args.unit_order == "lr":
            xc = [(s["bbox_person"][0] + s["bbox_person"][2]) / 2 for s in students]
            unit_order = np.argsort(xc, kind="stable")
        else:
            unit_order = conf_order
        units = [StudentUnit(index=k + 1,
                             text=students[i]["caption"],
                             action=students[i]["labels"]["activity"],
                             emotion=students[i]["labels"]["engagement_level"])
                 for k, i in enumerate(unit_order)]
        # true presented-box index for each presented unit
        pos_in_conf = np.empty(n, dtype=int)
        pos_in_conf[conf_order] = np.arange(n)
        true_box_of_unit = pos_in_conf[unit_order]

        crops = [Image.open(LLMSTU_ROOT / "crops" /
                            students[i]["crop_file"]).convert("RGB")
                 for i in conf_order]
        n_crops += n
        clip_sim = scorer.similarity_matrix([u.text for u in units], crops)
        spatial = spatial_agreement([u.index for u in units], boxes)
        cost = build_cost_matrix(clip_sim, spatial, det_conf, weights)

        results = {
            "ordinal": match_ordinal(n, n),
            "ordinal_lr": match_ordinal_lr(n, boxes),
            "hungarian": match_hungarian(cost),
            "sinkhorn": match_sinkhorn(cost, eps=args.eps)[0],
        }
        for strat, pairs in results.items():
            evaluators[strat].add_frame(pairs, true_box_of_unit, boxes, n)

        if args.dump is not None:
            comps = assignment_components(results["hungarian"], clip_sim,
                                          spatial, det_conf, cost, units)
            for c in comps:
                c["correct"] = bool(true_box_of_unit[c["unit_idx"]] == c["box_idx"])
                c["src_frame"] = frame["src_frame"]
                dump_records.append(c)

        if (fi + 1) % 50 == 0:
            rate = n_crops / (time.time() - t0)
            print(f"  {fi + 1}/{len(gt)} frames  ({rate:.1f} crops/s)")

    summaries = {s: evaluators[s].summary() for s in STRATEGIES}
    print(f"\nunit-order={args.unit_order}  spatial-weight={args.spatial_weight}"
          f"  device={args.device}  elapsed={time.time() - t0:.0f}s")
    print(format_results_table(summaries))

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump, "w") as fh:
            for rec in dump_records:
                fh.write(json.dumps(rec) + "\n")
        print(f"\nwrote {len(dump_records)} assignments -> {args.dump}")

    out = {"config": vars(args) | {"gt": str(args.gt), "standin": used_standin,
                                   "dump": str(args.dump), "standin_shard": str(args.standin_shard)},
           "results": summaries}
    out_path = Path("work_dirs/matching") / \
        f"results_{args.unit_order}_sw{args.spatial_weight}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"results json -> {out_path}")


if __name__ == "__main__":
    main()
