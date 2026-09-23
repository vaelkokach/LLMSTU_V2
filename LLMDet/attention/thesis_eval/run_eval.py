"""Authoritative single-process evaluation of one temporal checkpoint.

Produces, for one (checkpoint, split) pair:

    metrics.json          full frame-level + calibration block, with
                          video-level cluster-bootstrap 95% CIs
    per_class.csv         per-class precision/recall/F1/AUPRC/AUROC/support
    confusion_matrix.csv  rows = truth, columns = prediction
    reliability.csv       confidence-vs-accuracy bins
    predictions.npz       probabilities + targets + video/seat/time metadata,
                          so every metric above can be recomputed, and
                          calibration or abstention re-tuned, WITHOUT rerunning
                          the model

The prediction archive is the point: this project has repeatedly had to rerun
inference to answer a question a stored probability array would have answered,
and twice compared numbers produced by different harnesses because rerunning
was expensive.

Usage (from LLMDet/):
    python -m attention.thesis_eval.run_eval \
        --ckpt work_dirs/thesis/ladder/570_full_s42/checkpoints/best.pth \
        --split val --out work_dirs/thesis/ladder/570_full_s42/eval_val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import bootstrap as B
from attention.thesis_eval import data as D
from attention.thesis_eval import metrics as M
from attention.thesis_eval.models import build_model


def load_checkpoint(path: Path):
    ck = torch.load(path, map_location="cpu")
    spec = ck.get("spec")
    if spec is None:
        raise SystemExit(
            f"{path} carries no 'spec'; it predates the unified trainer. "
            "Re-train with attention.thesis_eval.train, or evaluate it with the "
            "legacy harness and label the number as legacy.")
    return ck, spec


@torch.no_grad()
def predict(model, seqs: List[D.Sequence_], device, batch_size: int = 1,
            use_refine: bool = False) -> Dict[str, np.ndarray]:
    """Run the model over sequences and return flat per-frame arrays.

    ``batch_size=1`` is the default because it needs no padding at all; larger
    batches exercise the padding mask and must give identical results, which
    ``tests/test_thesis_eval.py`` asserts.
    """
    model.eval()
    probs, preds, tgts, vids, seats, times, keys = [], [], [], [], [], [], []
    order = list(range(len(seqs)))
    for i in range(0, len(order), batch_size):
        chunk = [seqs[j] for j in order[i:i + batch_size]]
        x, y, mask = D.collate([(s.x, s.y) for s in chunk])
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        out = model(x, pad_mask=mask)
        p = torch.softmax(out["logits"].float(), dim=-1)
        if use_refine and "boundary" in out:
            lengths = (~mask).sum(1)
            refined = model.refine(out["logits"].float(), out["boundary"].float(), lengths)
        else:
            refined = p.argmax(-1)
        for bi, s in enumerate(chunk):
            T = s.x.shape[0]
            probs.append(p[bi, :T].cpu().numpy())
            preds.append(refined[bi, :T].cpu().numpy())
            tgts.append(s.y)
            vids.extend([s.video_id] * T)
            seats.extend([s.seat_id] * T)
            times.append(s.t)
            keys.extend([s.key] * T)
    arr = {
        "probs": np.concatenate(probs).astype(np.float32),
        "pred": np.concatenate(preds).astype(np.int64),
        "y": np.concatenate(tgts).astype(np.int64),
        "video_id": np.array(vids), "seat_id": np.array(seats, dtype=np.int64),
        "t": np.concatenate(times), "seq_key": np.array(keys),
    }
    # Drop the frames the taxonomy abstains on, exactly as the training loop's
    # evaluate_logits does (`valid = y != IGNORE`). Without this an abstained
    # label reaches brier_score as index -100 and raises IndexError; worse, a
    # metric that happened not to index by label would have silently averaged
    # over frames the model was never asked to predict.
    #
    # Filtered here rather than in main() so that metrics, the cluster
    # bootstrap and the saved predictions.npz all describe the same frames.
    # For cue6 nothing is dropped and every existing archive is unchanged.
    keep = arr["y"] != D.IGNORE_INDEX
    if not keep.all():
        arr = {k: v[keep] for k, v in arr.items()}
    return arr


def add_bootstrap(res: Dict, arr: Dict, cluster: str = "video",
                  n_boot: int = 2000, seed: int = 0) -> Dict:
    """Video-level (default) or track-level cluster bootstrap on the headline
    metrics. Frame-level resampling is deliberately not offered."""
    ids = (arr["video_id"] if cluster == "video"
           else np.char.add(arr["video_id"], arr["seat_id"].astype(str)))
    y, pred, probs = arr["y"], arr["pred"], arr["probs"]

    def cm_stats(idx):
        """All confusion-matrix metrics from a single CM per resample."""
        cm = M.confusion_matrix(y[idx], pred[idx])
        prf = M.per_class_prf(cm)
        out = {"accuracy": M.accuracy(cm),
               "balanced_accuracy": M.balanced_accuracy(cm),
               "macro_f1": M.macro_f1(cm),
               "weighted_f1": M.weighted_f1(cm)}
        for c, name in enumerate(CUE_CLASSES):
            out[f"f1::{name}"] = float(prf["f1"][c])
            out[f"recall::{name}"] = float(prf["recall"][c])
        return out

    out = B.cluster_bootstrap_multi(ids, cm_stats, n_boot, seed=seed)

    # Ranking and calibration need the probability array and are markedly more
    # expensive per resample, so they get fewer draws — stated in the output
    # rather than silently mixed with the cheaper ones.
    n_rank = max(200, n_boot // 4)

    def prob_stats(idx):
        r = M.ranking_metrics(probs[idx], y[idx])
        v = [x for x in r["auprc"] if np.isfinite(x)]
        return {"macro_auprc": float(np.mean(v)) if v else float("nan"),
                "ece": M.expected_calibration_error(probs[idx], y[idx]),
                "brier": M.brier_score(probs[idx], y[idx]),
                "nll": M.negative_log_likelihood(probs[idx], y[idx])}

    out.update(B.cluster_bootstrap_multi(ids, prob_stats, n_rank, seed=seed))
    res["bootstrap"] = {"cluster_unit": cluster, "n_boot_cm": n_boot,
                        "n_boot_probability_metrics": n_rank, **out}
    return res


def write_tables(out_dir: Path, res: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "auprc", "auroc", "support",
                    "f1_ci_low", "f1_ci_high"])
        for name in res["class_order"]:
            m = res["per_class"][name]
            ci = res.get("bootstrap", {}).get(f"f1::{name}", {})
            w.writerow([name, f"{m['precision']:.6f}", f"{m['recall']:.6f}",
                        f"{m['f1']:.6f}", f"{m['auprc']:.6f}", f"{m['auroc']:.6f}",
                        m["support"], f"{ci.get('ci_low', float('nan')):.6f}",
                        f"{ci.get('ci_high', float('nan')):.6f}"])
    cm = np.array(res["confusion_matrix"])
    with open(out_dir / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["truth\\pred"] + list(res["class_order"]))
        for i, name in enumerate(res["class_order"]):
            w.writerow([name] + cm[i].tolist())
    with open(out_dir / "reliability.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_lo", "bin_hi", "count", "mean_confidence", "accuracy"])
        for r in res["reliability"]:
            w.writerow([r["bin_lo"], r["bin_hi"], r["count"],
                        r["mean_confidence"], r["accuracy"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cluster", default="video", choices=["video", "track"])
    ap.add_argument("--refine", action="store_true",
                    help="ASRF only: relabel by predicted boundaries")
    ap.add_argument("--no-bootstrap", action="store_true")
    # Default to whatever the CHECKPOINT was trained on. Passing the wrong
    # sequence root is otherwise easy and its symptom is a layout error far
    # from the cause -- or, for two builds of the same width, no error at all.
    ap.add_argument("--manifest", default=None,
                    help="default: the manifest recorded in the checkpoint")
    ap.add_argument("--cue-labels", default=None,
                    help="override the cue-label set. Defaults to the one the "
                         "checkpoint was TRAINED on (spec.cue_labels), so a "
                         "model is scored against its own target unless you "
                         "deliberately cross-evaluate. Pass '' to force the "
                         "labels stored in the sequences.")
    ap.add_argument("--sequence-root", default=None,
                    help="default: the sequence root recorded in the checkpoint")
    args = ap.parse_args()

    if args.split == "test":
        print("*** TEST SPLIT — protocol-gated. This must be run once, after the "
              "model and evaluation protocol are frozen. ***", flush=True)

    ck, spec = load_checkpoint(Path(args.ckpt))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # The taxonomy comes from the CHECKPOINT's own spec, never from a flag:
    # evaluating a coarse-trained model against the 6-class labels would score
    # it on a task it was not trained for and silently report nonsense.
    from attention.taxonomy import taxonomy_classes
    taxonomy = spec.get("taxonomy", "cue6")
    class_names = taxonomy_classes(taxonomy)
    manifest = args.manifest or spec.get(
        "manifest", "../grounding_data/llmstu_seq_split_manifest.json")
    seq_root = args.sequence_root or spec.get(
        "sequence_root", "../grounding_data/llmstu_sequences_full")
    print(f"[eval] sequences {seq_root} | manifest {manifest} | "
          f"config {spec['feature_config']} | taxonomy {taxonomy}", flush=True)
    trained_on = spec.get("cue_labels", "") or ""
    cue_labels = trained_on if args.cue_labels is None else args.cue_labels
    if cue_labels != trained_on:
        print(f"  CROSS-EVALUATION: checkpoint was trained against "
              f"{trained_on or '<labels stored in the sequences>'} and is being "
              f"scored against {cue_labels or '<labels stored in the sequences>'}. "
              f"This is a different target; say so wherever the number is used.",
              flush=True)
    seqs = D.load_split(Path(manifest), Path(seq_root),
                        args.split, spec["feature_config"], taxonomy=taxonomy,
                        cue_labels=Path(cue_labels) if cue_labels else None)
    dim = D.config_dim(spec["feature_config"])
    kw = dict(spec.get("model_kwargs") or {})
    if spec["model"] == "transformer":
        kw.setdefault("dropout", spec.get("dropout", 0.1))
    model = build_model(spec["model"], dim, len(class_names), **kw).to(device)
    model.load_state_dict(ck["model"])

    n_total = int(sum(len(s.y) for s in seqs))
    arr = predict(model, seqs, device, args.batch_size, use_refine=args.refine)
    res = M.frame_metrics(arr["probs"], arr["y"], arr["pred"],
                          num_classes=len(class_names),
                          class_names=class_names)
    res.update({
        "taxonomy": taxonomy,
        "class_order": class_names,
        "evaluator_version": EVALUATOR_VERSION,
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "experiment_id": spec.get("experiment_id"),
        "cue_labels": cue_labels, "cue_labels_trained_on": trained_on,
        "model": spec["model"], "feature_config": spec["feature_config"],
        "input_dim": dim, "seed": spec.get("seed"),
        "split": args.split, "n_sequences": len(seqs),
        # Coverage travels with the number. An abstaining taxonomy scores a
        # subset of frames, and a macro-F1 quoted without saying which subset
        # is not interpretable.
        "n_frames_total": n_total,
        "n_frames_abstained": n_total - int(len(arr["y"])),
        "coverage": float(len(arr["y"]) / max(n_total, 1)),
        "n_videos": int(len(set(arr["video_id"]))),
        "batch_size": args.batch_size, "refined": bool(args.refine),
    })
    if not args.no_bootstrap:
        add_bootstrap(res, arr, args.cluster, args.n_boot)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=2))
    write_tables(out_dir, res)
    np.savez_compressed(out_dir / "predictions.npz", **arr)
    (out_dir / "command.txt").write_text(
        "python -m attention.thesis_eval.run_eval "
        f"--ckpt {args.ckpt} --split {args.split} --out {args.out} "
        f"--sequence-root {seq_root} --manifest {manifest} "
        f"--batch-size {args.batch_size} --cluster {args.cluster}"
        + (" --refine" if args.refine else "") + "\n")

    b = res.get("bootstrap", {})
    ci = lambda k: (f" [{b[k]['ci_low']:.4f}, {b[k]['ci_high']:.4f}]" if k in b else "")
    print(f"{res['experiment_id']} / {args.split}: "
          f"acc {res['accuracy']:.4f}{ci('accuracy')}  "
          f"bal_acc {res['balanced_accuracy']:.4f}{ci('balanced_accuracy')}  "
          f"macroF1 {res['macro_f1']:.4f}{ci('macro_f1')}  "
          f"mAUPRC {res['macro_auprc']:.4f}  ECE {res['ece']:.4f}  "
          f"coverage {res['coverage'] * 100:.1f}% "
          f"({res['n_frames']}/{res['n_frames_total']} frames)")


if __name__ == "__main__":
    main()
