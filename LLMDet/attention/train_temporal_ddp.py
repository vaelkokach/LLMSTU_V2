import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from attention.temporal_model import AttentionTransformer

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    average_precision_score = None

IGNORE_INDEX = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceNPZDataset(Dataset):
    """NPZ sequences with per-frame labels (``y_frames``); older files with
    only a scalar ``y`` are broadcast across all frames."""

    def __init__(self, seq_dir: Path, label_remap: dict | None = None):
        self.files = sorted(seq_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {seq_dir}")
        self._remap_lut = None
        if label_remap:
            lut = np.arange(max(label_remap) + 1, dtype=np.int64)
            for src, dst in label_remap.items():
                lut[src] = dst
            self._remap_lut = lut

    def _remap(self, y: np.ndarray) -> np.ndarray:
        if self._remap_lut is None:
            return y
        valid = y != IGNORE_INDEX
        y = y.copy()
        y[valid] = self._remap_lut[np.clip(y[valid], 0, len(self._remap_lut) - 1)]
        return y

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        item = np.load(self.files[idx])
        x = item["x"].astype(np.float32)
        if "y_frames" in item:
            y = item["y_frames"].astype(np.int64)
        elif "y" in item:
            y = np.full((x.shape[0],), int(item["y"].item()), dtype=np.int64)
        else:
            y = np.full((x.shape[0],), IGNORE_INDEX, dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(self._remap(y))

    def label_histogram(self, num_classes: int) -> np.ndarray:
        hist = np.zeros((num_classes,), dtype=np.int64)
        for fp in self.files:
            item = np.load(fp)
            if "y_frames" in item:
                ys = item["y_frames"].astype(np.int64)
            elif "y" in item:
                ys = np.array([int(item["y"].item())], dtype=np.int64)
            else:
                continue
            ys = self._remap(ys)
            for y in ys.tolist():
                if 0 <= y < num_classes:
                    hist[y] += 1
        return hist


def collate_varlen(batch):
    """Zero-pad features, IGNORE_INDEX-pad labels, and return a padding mask
    (True at padded positions)."""
    xs, ys = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    d = xs[0].shape[1]
    out = torch.zeros((len(xs), max_t, d), dtype=torch.float32)
    lab = torch.full((len(xs), max_t), IGNORE_INDEX, dtype=torch.long)
    mask = torch.ones((len(xs), max_t), dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        out[i, : x.shape[0]] = x
        lab[i, : y.shape[0]] = y
        mask[i, : x.shape[0]] = False
    return out, lab, mask


def init_ddp(launcher: str) -> Tuple[bool, int, int, int]:
    if launcher == "none":
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="DDP temporal training for student attention cues.")
    p.add_argument("--config", type=str, required=True, help="YAML config path.")
    p.add_argument("--launcher", type=str, default="none", choices=["none", "pytorch"])
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target.tolist(), pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, List[float]]:
    num_classes = cm.shape[0]
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f = 2.0 * p * r / (p + r + 1e-6)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(cm: np.ndarray, present_only: bool = True) -> float:
    """Macro-F1; by default averaged over classes that actually appear in the
    targets, so absent classes don't silently deflate/inflate the score."""
    cls = per_class_metrics(cm)
    support = cm.sum(axis=1)
    f1s = [f for f, s in zip(cls["f1"], support) if (s > 0 or not present_only)]
    return float(np.mean(f1s)) if f1s else 0.0


def auprc_per_class(probs: np.ndarray, targets: np.ndarray, num_classes: int) -> List[float]:
    """One-vs-rest average precision per class; NaN for absent classes."""
    out = []
    for c in range(num_classes):
        pos = (targets == c).astype(np.int64)
        if pos.sum() == 0:
            out.append(float("nan"))
            continue
        if average_precision_score is not None:
            out.append(float(average_precision_score(pos, probs[:, c])))
        else:
            order = np.argsort(-probs[:, c])
            pos_sorted = pos[order]
            tp = np.cumsum(pos_sorted)
            prec = tp / (np.arange(len(pos_sorted)) + 1)
            out.append(float((prec * pos_sorted).sum() / max(1, pos_sorted.sum())))
    return out


def _masked_ce(logits: torch.Tensor, y: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over [B, T, C] logits with IGNORE_INDEX-padded targets."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        weight=class_weights,
        ignore_index=IGNORE_INDEX,
    )


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, class_weights: torch.Tensor):
    model.train()
    losses = []
    correct, total = 0, 0

    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg["training"].get("amp", True) and device.type == "cuda"):
            logits = model(x, key_padding_mask=mask)  # [B, T, C]
            valid = y != IGNORE_INDEX
            if not valid.any():
                continue
            loss = _masked_ce(logits, y, class_weights)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        pred = logits.argmax(dim=-1)
        correct += int((pred[valid] == y[valid]).sum().item())
        total += int(valid.sum().item())
    acc = correct / total if total else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc


@torch.no_grad()
def validate(model, loader, device, class_weights: torch.Tensor, num_classes: int):
    model.eval()
    losses = []
    correct, total = 0, 0
    all_pred, all_tgt, all_prob = [], [], []
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        valid = y != IGNORE_INDEX
        if not valid.any():
            continue
        logits = model(x, key_padding_mask=mask)
        loss = _masked_ce(logits, y, class_weights)
        probs = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)
        all_pred.append(pred[valid].detach().cpu().numpy())
        all_tgt.append(y[valid].detach().cpu().numpy())
        all_prob.append(probs[valid].detach().float().cpu().numpy())
        losses.append(float(loss.item()))
        correct += int((pred[valid] == y[valid]).sum().item())
        total += int(valid.sum().item())
    if all_pred:
        pred_np = np.concatenate(all_pred, axis=0)
        tgt_np = np.concatenate(all_tgt, axis=0)
        prob_np = np.concatenate(all_prob, axis=0)
        cm = compute_confusion_matrix(pred_np, tgt_np, num_classes=num_classes)
        cls = per_class_metrics(cm)
        cls["auprc"] = auprc_per_class(prob_np, tgt_np, num_classes)
        cls["macro_f1"] = macro_f1(cm)
        finite_ap = [a for a in cls["auprc"] if not np.isnan(a)]
        cls["macro_auprc"] = float(np.mean(finite_ap)) if finite_ap else 0.0
    else:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        cls = {
            "precision": [0.0] * num_classes,
            "recall": [0.0] * num_classes,
            "f1": [0.0] * num_classes,
            "auprc": [float("nan")] * num_classes,
            "macro_f1": 0.0,
            "macro_auprc": 0.0,
        }
    acc = correct / total if total else 0.0
    return float(np.mean(losses)) if losses else 0.0, acc, cm, cls


def main():
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg.get("seed", 42)))

    ddp, rank, _, local_rank = init_ddp(args.launcher)
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seq_root = Path(cfg["data"]["sequence_dir"])
    num_classes = int(cfg["model"]["num_classes"])
    # Sequences built with the old 7-class taxonomy (idle_other=5, uncertain=6)
    # fold into the current 6-class one at load time.
    from attention.taxonomy import LEGACY_7CLASS_REMAP, NUM_CUE_CLASSES
    label_remap = LEGACY_7CLASS_REMAP if num_classes == NUM_CUE_CLASSES == 6 else None
    train_set = SequenceNPZDataset(seq_root / "train", label_remap=label_remap)
    val_set = SequenceNPZDataset(seq_root / "val", label_remap=label_remap)
    use_class_weights = bool(cfg["training"].get("use_class_weights", True))
    class_hist = train_set.label_histogram(num_classes)
    val_hist = val_set.label_histogram(num_classes)
    required_min_per_class = int(cfg["training"].get("required_min_per_class", 1))
    enforce_full_class_coverage = bool(cfg["training"].get("enforce_full_class_coverage", True))
    missing = [i for i, c in enumerate(class_hist.tolist()) if c < required_min_per_class]
    if enforce_full_class_coverage and missing:
        raise RuntimeError(
            f"Insufficient class coverage in training data. class_hist={class_hist.tolist()}, "
            f"required_min_per_class={required_min_per_class}, missing_class_indices={missing}. "
            "Add/curate data for missing classes before training."
        )
    if use_class_weights:
        # Square-root inverse-frequency weighting: mitigates dominant-class
        # collapse without letting a tiny class blow up the loss (plain
        # inverse-frequency gave a ~10,000x weight ratio and destabilized
        # training when one class had 16 samples).
        safe = np.maximum(class_hist.astype(np.float32), 1.0)
        inv = 1.0 / np.sqrt(safe)
        inv = inv / np.sum(inv) * num_classes
        class_weights_np = inv.astype(np.float32)
    else:
        class_weights_np = np.ones((num_classes,), dtype=np.float32)
    class_weights = torch.from_numpy(class_weights_np).to(device)
    if is_main:
        print(
            f"[info] train class_hist={class_hist.tolist()} val class_hist={val_hist.tolist()} "
            f"class_weights={class_weights_np.tolist()}"
        )

    train_sampler = DistributedSampler(train_set, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_varlen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_varlen,
    )

    model = AttentionTransformer(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
        per_frame=bool(cfg["model"].get("per_frame", True)),
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")

    out_dir = Path(cfg["training"]["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    tb_dir = out_dir / "tb"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir)) if is_main else None

    # Model selection on macro-F1, not accuracy: with heavy class imbalance
    # accuracy rewards majority-class collapse.
    best_val = -1.0
    best_val_acc = -1.0
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, class_weights)
        va_loss, va_acc, va_cm, va_cls = validate(model, val_loader, device, class_weights, num_classes)

        if is_main:
            print(
                f"[epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                f"val_macro_f1={va_cls['macro_f1']:.4f} val_macro_auprc={va_cls['macro_auprc']:.4f}"
            )
            if writer is not None:
                writer.add_scalar("train/loss", tr_loss, epoch)
                writer.add_scalar("train/acc", tr_acc, epoch)
                writer.add_scalar("val/loss", va_loss, epoch)
                writer.add_scalar("val/acc", va_acc, epoch)
                writer.add_scalar("val/macro_f1", float(va_cls["macro_f1"]), epoch)
                writer.add_scalar("val/macro_auprc", float(va_cls["macro_auprc"]), epoch)
                for c in range(num_classes):
                    writer.add_scalar(f"val/class_{c}_precision", float(va_cls["precision"][c]), epoch)
                    writer.add_scalar(f"val/class_{c}_recall", float(va_cls["recall"][c]), epoch)
                    writer.add_scalar(f"val/class_{c}_f1", float(va_cls["f1"][c]), epoch)

            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")
            if va_cls["macro_f1"] > best_val:
                best_val = float(va_cls["macro_f1"])
                best_val_acc = va_acc
                torch.save(state, ckpt_dir / "best.pth")
                np.save(ckpt_dir / "best_val_confusion_matrix.npy", va_cm)
                with (ckpt_dir / "best_val_class_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(va_cls, f, indent=2)

    if is_main and writer is not None:
        writer.close()
        with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_macro_f1": best_val,
                    "best_val_acc": best_val_acc,
                    "train_class_hist": class_hist.tolist(),
                    "val_class_hist": val_hist.tolist(),
                    "class_weights": class_weights_np.tolist(),
                },
                f,
                indent=2,
            )

    cleanup_ddp()


if __name__ == "__main__":
    main()
