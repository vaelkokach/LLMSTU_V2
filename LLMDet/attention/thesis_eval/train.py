"""Single-process trainer for the controlled cue-model experiments.

Deliberately **not** DDP. These models train in minutes on one A100, and DDP
bought this project nothing but two rounds of shard-averaged validation metrics
that misordered the ablation ladder [internal notes, not included], [internal notes, not included]. Independent
single-GPU runs are launched in parallel across GPUs instead, which is both
faster in wall clock for a 15-run sweep and free of any cross-rank reduction
to get wrong.

Everything that could confound the ablation is pinned by ``ExperimentSpec`` and
written verbatim into the run record: split manifest hash, feature columns,
seed, optimiser, schedule, batch size, update count and the
checkpoint-selection rule. The only thing a ladder rung may change is
``feature_config`` (or ``model`` for the architecture comparison).

Checkpoint selection
--------------------
Validation macro-F1 in this project swings ~0.05 between adjacent epochs, so
taking the argmax over 90 epochs selects partly on noise and biases the
reported best upward. Selection here uses the **mean of the last-k epochs'
macro-F1** to pick an epoch (``--select-window``, default 5), which is a
smoothed criterion, and both the selected and final checkpoints are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from attention.taxonomy import CUE_CLASSES
from attention.thesis_eval import EVALUATOR_VERSION
from attention.thesis_eval import data as D
from attention.thesis_eval import metrics as M
from attention.thesis_eval.models import boundary_targets_from_labels, build_model

IGNORE = D.IGNORE_INDEX


class SeqDataset(Dataset):
    def __init__(self, seqs: List[D.Sequence_]):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        s = self.seqs[i]
        return s.x, s.y


@dataclass
class ExperimentSpec:
    experiment_id: str
    model: str = "transformer"
    feature_config: str = "570_full"
    taxonomy: str = "cue6"
    partial_labels: bool = False
    seed: int = 42
    epochs: int = 90
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    amp: bool = True
    select_window: int = 5
    use_class_weights: bool = True
    boundary_loss_weight: float = 1.0
    smoothing_loss_weight: float = 0.15
    manifest: str = "../grounding_data/llmstu_seq_split_manifest.json"
    sequence_root: str = "../grounding_data/llmstu_sequences_full"
    #: npz from build_cue_labels: recomputed 6-class labels under a different
    #: cue RULE version. Empty means the labels stored in the sequences. Part
    #: of the spec, so it is written into every checkpoint and run record --
    #: a model's target set must travel with it, or a later evaluation will
    #: score it against a target it was never trained on.
    cue_labels: str = ""
    output_dir: str = ""
    model_kwargs: Dict = field(default_factory=dict)


def git_state() -> Dict[str, str]:
    def run(*a):
        try:
            return subprocess.check_output(a, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "unknown"
    return {"commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain")),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD")}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_threads(n: int = 8) -> None:
    """Cap intra-op threads.

    The sweep runs one process per GPU; left alone each would claim all 128
    cores, and the resulting oversubscription drove GPU utilisation to ~3% and
    epoch time from 6 s to 100 s. These models are small enough that the host
    side is the bottleneck, so this is the single most important knob.
    """
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
    torch.set_num_threads(n)


def _smoothing_loss(logits: torch.Tensor, valid: torch.Tensor,
                    clamp: float = 16.0) -> torch.Tensor:
    """Truncated MSE between adjacent log-probabilities (MS-TCN's T-MSE).

    This is the term that actually suppresses over-segmentation; without it a
    multi-stage TCN fragments badly, which would make an architecture
    comparison against the transformer meaningless.
    """
    logp = F.log_softmax(logits, dim=-1)
    d = (logp[:, 1:] - logp[:, :-1]) ** 2
    d = torch.clamp(d, max=clamp)
    m = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1).to(d.dtype)
    return (d * m).sum() / m.sum().clamp(min=1.0) / logits.size(-1)


def proden_loss(logits, cand, valid, weight=None):
    """Partial-label cross-entropy (PRODEN, Lv et al., ICML 2020).

    The labels are not single classes. `map_record` fires several cue rules on
    13.7% of LLMSTU records and keeps the highest-precedence one; measured over
    the pseudo-labels, `looking_away` is true by its own rule 2.74x more often
    than precedence lets it be the label. Standard cross-entropy then actively
    penalises the model for predicting a class the annotation itself supports.

    PRODEN replaces the one-hot target with the candidate set and lets the
    model decide which member to commit to:

        L = - sum_{c in S} w_c log p_c ,   w_c = p_c / sum_{c' in S} p_c'

    with ``w`` detached, so it is a target rather than a path for gradients.
    The weights start uniform over S (the network is near-uniform at init) and
    sharpen as the model becomes able to tell the candidates apart -- the
    "progressive identification" of the name. Where |S| = 1 this is exactly
    cross-entropy, so frames with an unambiguous label are unaffected.

    ``weight`` applies the same per-class weighting as the single-label path,
    so the two objectives stay comparable.
    """
    logp = torch.log_softmax(logits.float(), dim=-1)
    with torch.no_grad():
        w = torch.softmax(logits.float(), dim=-1) * cand
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
    per_class = -(w * logp)
    v = valid.reshape(-1).float()

    if weight is None:
        denom = v.sum().clamp_min(1.0)
    else:
        per_class = per_class * weight.view(1, -1)
        # Match F.cross_entropy(weight=...)'s reduction exactly. Torch's
        # weighted mean divides by the SUM OF TARGET WEIGHTS, not by the frame
        # count. Dividing by N instead made this loss 2.21x smaller than the
        # single-label path on the real class histogram, so the partial-label
        # runs trained with a proportionally weaker classification gradient
        # while the boundary and smoothing terms kept full strength -- and the
        # comparison measured that, not PRODEN.
        #
        # For a singleton candidate set w is one-hot, so this is exactly
        # sum(weight[y]) and the whole expression collapses to torch's.
        denom = ((w * weight.view(1, -1)).sum(-1) * v).sum().clamp_min(1e-12)

    return ((per_class.sum(-1)) * v).sum() / denom


@torch.no_grad()
def evaluate_logits(model, X, Y, mask, batch_size: int, num_classes: int):
    """Collect probabilities and targets over a whole split, single process.

    The whole split at once — never a running mean of per-batch metrics. That
    error is why the March accuracy read 2.6 points high and why the DDP
    trainer's per-shard macro-F1 misordered the ablation ladder.
    """
    model.eval()
    probs, tgts = [], []
    for i in range(0, X.shape[0], batch_size):
        x, y, m = X[i:i + batch_size], Y[i:i + batch_size], mask[i:i + batch_size]
        out = model(x, pad_mask=m)
        valid = y != IGNORE
        p = torch.softmax(out["logits"].float(), dim=-1)
        probs.append(p[valid].cpu().numpy())
        tgts.append(y[valid].cpu().numpy())
    return np.concatenate(probs), np.concatenate(tgts)


def train(spec: ExperimentSpec, device_str: str = "cuda:0", threads: int = 8) -> Dict:
    t0 = time.time()
    limit_threads(threads)
    set_seed(spec.seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    manifest = Path(spec.manifest)
    root = Path(spec.sequence_root)

    from attention.taxonomy import taxonomy_classes, taxonomy_excluded
    class_names = taxonomy_classes(spec.taxonomy)
    num_classes = len(class_names)
    cue_labels = Path(spec.cue_labels) if spec.cue_labels else None
    train_seqs = D.load_split(manifest, root, "train", spec.feature_config,
                              taxonomy=spec.taxonomy, cue_labels=cue_labels)
    val_seqs = D.load_split(manifest, root, "val", spec.feature_config,
                            taxonomy=spec.taxonomy, cue_labels=cue_labels)
    if cue_labels is not None:
        print(f"[{spec.experiment_id}] cue labels overridden from {cue_labels} "
              f"(ruleset {D.CueLabels(cue_labels).ruleset!r}) -- macro-F1 from "
              f"this run is measured against a DIFFERENT target than a run "
              f"without it, and the two are not directly comparable.", flush=True)
    hist = D.class_histogram(train_seqs, num_classes)
    val_hist = D.class_histogram(val_seqs, num_classes)
    if spec.taxonomy != "cue6":
        kept = int(hist.sum()) + int(val_hist.sum())
        raw = sum(int((s.y != D.IGNORE_INDEX).sum()) + int((s.y == D.IGNORE_INDEX).sum())
                  for s in train_seqs + val_seqs)
        print(f"[{spec.experiment_id}] taxonomy {spec.taxonomy}: {class_names}, "
              f"abstaining on {taxonomy_excluded(spec.taxonomy)} -> "
              f"{kept}/{raw} frames scored ({kept / max(raw, 1) * 100:.1f}% coverage)",
              flush=True)
    weights = (torch.from_numpy(D.sqrt_inverse_frequency_weights(hist)).to(device)
               if spec.use_class_weights else None)

    dim = train_seqs[0].x.shape[1]
    assert dim == D.config_dim(spec.feature_config), "feature slice width mismatch"

    # Pad once, keep resident on the GPU; a training step is then a tensor
    # slice. See data.to_padded_tensors for why.
    Xtr, Ytr, Mtr = D.to_padded_tensors(train_seqs, device)
    Xva, Yva, Mva = D.to_padded_tensors(val_seqs, device)
    Ctr = (D.padded_candidates(train_seqs, num_classes, device, Xtr.shape[1])
           if spec.partial_labels else None)
    if spec.partial_labels:
        amb = float((Ctr.sum(-1) > 1).float()[Ytr != IGNORE].mean())
        print(f"[{spec.experiment_id}] partial labels on: "
              f"{amb * 100:.1f}% of scored frames carry >1 candidate", flush=True)
    g = torch.Generator(device="cpu"); g.manual_seed(spec.seed)

    kw = dict(spec.model_kwargs)
    if spec.model == "transformer":
        kw.setdefault("dropout", spec.dropout)
    model = build_model(spec.model, dim, num_classes, **kw).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=spec.amp and device.type == "cuda")

    out_dir = Path(spec.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    history: List[Dict] = []
    updates = 0
    for epoch in range(spec.epochs):
        model.train()
        losses = []
        perm = torch.randperm(Xtr.shape[0], generator=g).to(device)
        for bi in range(0, Xtr.shape[0], spec.batch_size):
            idx = perm[bi:bi + spec.batch_size]
            x, y, mask = Xtr[idx], Ytr[idx], Mtr[idx]
            cand = Ctr[idx] if Ctr is not None else None
            valid = y != IGNORE
            if not valid.any():
                continue
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=spec.amp and device.type == "cuda"):
                out = model(x, pad_mask=mask)
                stages = out.get("aux_logits")
                if stages is None:
                    stages = out["logits"].unsqueeze(0)
                loss = 0.0
                for s in stages:
                    if cand is None:
                        loss = loss + F.cross_entropy(
                            s.reshape(-1, num_classes), y.reshape(-1),
                            weight=weights, ignore_index=IGNORE)
                    else:
                        loss = loss + proden_loss(
                            s.reshape(-1, num_classes),
                            cand.reshape(-1, num_classes), valid, weights)
                    if spec.smoothing_loss_weight > 0:
                        loss = loss + spec.smoothing_loss_weight * _smoothing_loss(s, valid)
                if "boundary" in out:
                    bt = boundary_targets_from_labels(y, IGNORE)
                    bstages = out.get("aux_boundary", out["boundary"].unsqueeze(0))
                    vm = valid.to(bt.dtype)
                    # Transitions are ~2% of frames; unweighted BCE collapses to
                    # predicting "no boundary" everywhere, which would make the
                    # refinement step a no-op.
                    pos = bt.sum().clamp(min=1.0)
                    pw = ((vm.sum() - pos) / pos).clamp(1.0, 50.0)
                    for bs in bstages:
                        bl = F.binary_cross_entropy_with_logits(
                            bs, bt, reduction="none", pos_weight=pw)
                        loss = loss + spec.boundary_loss_weight * (
                            (bl * vm).sum() / vm.sum().clamp(min=1.0))
            scaler.scale(loss).backward()
            # NOTE — clip the *scaled* gradients, exactly as the historic
            # trainer did. This is not a no-op and it is not a bug to tidy:
            # clipping before unscaling forces the gradient vector to a fixed
            # norm every step, and because Adam is invariant to a constant
            # gradient rescaling the net effect is Adam on globally
            # **normalised** gradients. Measured over 20 epochs on 556_hp
            # (identical seed, data and schedule):
            # (measured value removed from the handover copy)
            # (measured value removed from the handover copy)
            # (measured value removed from the handover copy)
            # Both "corrected" variants collapse to the majority class. The
            # historic recipe is therefore reproduced deliberately, and the
            # retrained models stay comparable with the archived 552/556/570
            # checkpoints.
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
            updates += 1

        probs, tgts = evaluate_logits(model, Xva, Yva, Mva, spec.batch_size, num_classes)
        cm = M.confusion_matrix(tgts, probs.argmax(1), num_classes)  # IGNORE already dropped
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
               "val_accuracy": M.accuracy(cm), "val_macro_f1": M.macro_f1(cm),
               "val_balanced_accuracy": M.balanced_accuracy(cm)}
        history.append(row)
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "spec": asdict(spec)}, out_dir / "checkpoints" / "last.pth")
        # smoothed selection: mean macro-F1 over the trailing window
        w = spec.select_window
        smoothed = [float(np.mean([h["val_macro_f1"] for h in history[max(0, i - w + 1):i + 1]]))
                    for i in range(len(history))]
        row["smoothed_macro_f1"] = smoothed[-1]
        if smoothed[-1] >= max(smoothed):
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "spec": asdict(spec)}, out_dir / "checkpoints" / "best.pth")
        if epoch % 10 == 0 or epoch == spec.epochs - 1:
            print(f"[{spec.experiment_id}] ep {epoch:3d} loss {row['train_loss']:.4f} "
                  f"val_acc {row['val_accuracy']:.4f} macroF1 {row['val_macro_f1']:.4f} "
                  f"(smoothed {smoothed[-1]:.4f})", flush=True)

    best_epoch = int(np.argmax([h["smoothed_macro_f1"] for h in history]))
    record = {
        "experiment_id": spec.experiment_id,
        "evaluator_version": EVALUATOR_VERSION,
        "spec": asdict(spec),
        "git": git_state(),
        "manifest_sha256": file_hash(manifest),
        "cue_labels_sha256": file_hash(cue_labels) if cue_labels else "",
        "feature_columns": D.column_index(spec.feature_config).tolist(),
        "input_dim": int(dim),
        "n_parameters": int(n_params),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "n_train_sequences": len(train_seqs), "n_val_sequences": len(val_seqs),
        "n_train_frames": int(hist.sum()), "n_val_frames": int(val_hist.sum()),
        "train_class_hist": hist.tolist(), "val_class_hist": val_hist.tolist(),
        "class_weights": D.sqrt_inverse_frequency_weights(hist).tolist(),
        "total_updates": updates,
        "selection_rule": f"max mean(val_macro_f1) over trailing {spec.select_window} epochs",
        "selected_epoch": best_epoch,
        "selected_val_macro_f1": history[best_epoch]["val_macro_f1"],
        "final_val_macro_f1": history[-1]["val_macro_f1"],
        "wall_clock_s": time.time() - t0,
        "history": history,
    }
    (out_dir / "run_record.json").write_text(json.dumps(record, indent=2))
    print(f"[{spec.experiment_id}] done in {record['wall_clock_s']:.0f}s; "
          f"selected epoch {best_epoch} macroF1 {record['selected_val_macro_f1']:.4f}")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--feature-config", default="570_full", choices=sorted(D.FEATURE_CONFIGS))
    ap.add_argument("--partial-labels", action="store_true",
                    help="train with PRODEN over candidate sets instead of the "
                         "precedence-collapsed single label. Needs sequences "
                         "built with y_cand; older ones degrade to one-hot, "
                         "i.e. to plain cross-entropy.")
    ap.add_argument("--taxonomy", default="cue6",
                    help="label set: cue6 (default), onoff, onoff_reliable, "
                         "coarse3_reliable, cue9. The _reliable variants abstain "
                         "on looking_away and turned_to_peer; report coverage. "
                         "cue9 is a DIFFERENT label space (screen_oriented split "
                         "four ways) and REQUIRES --cue-labels from "
                         "`build_cue_labels --label-space cue9`.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--select-window", type=int, default=5)
    ap.add_argument("--smoothing-loss-weight", type=float, default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--manifest", default="../grounding_data/llmstu_seq_split_manifest.json")
    ap.add_argument("--sequence-root", default="../grounding_data/llmstu_sequences_full")
    ap.add_argument("--cue-labels", default="",
                    help="npz from attention.thesis_eval.build_cue_labels: "
                         "recomputed cue labels under a different rule version. "
                         "Features are unchanged; only the target differs.")
    args = ap.parse_args()

    # The transformer baseline must stay exactly as it was trained historically,
    # so the T-MSE smoothing term (an MS-TCN component) defaults off for it.
    smooth = args.smoothing_loss_weight
    if smooth is None:
        smooth = 0.0 if args.model == "transformer" else 0.15

    spec = ExperimentSpec(
        experiment_id=args.experiment_id, model=args.model,
        feature_config=args.feature_config, taxonomy=args.taxonomy,
        partial_labels=args.partial_labels,
        seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, select_window=args.select_window,
        smoothing_loss_weight=smooth, output_dir=args.output_dir,
        manifest=args.manifest, sequence_root=args.sequence_root,
        cue_labels=args.cue_labels)
    train(spec, args.device, args.threads)


if __name__ == "__main__":
    main()
