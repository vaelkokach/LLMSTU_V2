"""Box <-> student-unit compatibility scoring.

Produces the cost matrix consumed by the Hungarian/Sinkhorn matchers and the
raw per-pair components needed later for confidence calibration on the gold
calibration split (to-do item 28: don't hand-tune the confidence weighting,
calibrate it).

Components per (unit, box) pair:
- clip_sim   : CLIP ViT-B/32 image-text cosine similarity (crop vs unit text)
- spatial    : agreement between the unit's ordinal index and the box's
               left-to-right rank (VLM reading-order prior), in [0, 1]
- det_conf   : detector confidence of the box (unit-independent)
- fmt_valid  : unit parse validity (box-independent)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


@dataclass
class CompatibilityWeights:
    clip: float = 1.0
    spatial: float = 0.0   # off by default: the spatial prior is under test
    det_conf: float = 0.0


class ClipScorer:
    """CLIP ViT-B/32 image-text similarity. CPU by default; pass device="cuda"
    only with explicit approval for GPU use."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu", batch_size: int = 16):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = self.processor(images=list(images[i:i + self.batch_size]),
                                   return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**batch)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(feats)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        feats = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.processor(text=list(texts[i:i + self.batch_size]),
                                   return_tensors="pt", padding=True,
                                   truncation=True).to(self.device)
            f = self.model.get_text_features(**batch)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(feats)

    def similarity_matrix(self, texts: Sequence[str],
                          images: Sequence[Image.Image]) -> np.ndarray:
        """(N_texts, M_images) cosine similarities."""
        t = self.encode_texts(texts)
        v = self.encode_images(images)
        return (t @ v.T).cpu().numpy()


def spatial_agreement(unit_indices: Sequence[int],
                      boxes: np.ndarray) -> np.ndarray:
    """(N, M) score in [0,1]: 1 when unit ordinal rank equals the box's
    left-to-right rank, decaying linearly with rank distance."""
    n = len(unit_indices)
    xc = (boxes[:, 0] + boxes[:, 2]) / 2.0
    lr_rank = np.empty(len(xc), dtype=int)
    lr_rank[np.argsort(xc, kind="stable")] = np.arange(len(xc))
    unit_rank = np.argsort(np.argsort(unit_indices, kind="stable"))
    denom = max(max(n, len(xc)) - 1, 1)
    diff = np.abs(unit_rank[:, None] - lr_rank[None, :])
    return 1.0 - diff / denom


def build_cost_matrix(clip_sim: np.ndarray,
                      spatial: Optional[np.ndarray] = None,
                      det_conf: Optional[np.ndarray] = None,
                      weights: CompatibilityWeights = CompatibilityWeights(),
                      ) -> np.ndarray:
    """Combine components into a cost matrix (lower = better).

    clip_sim is z-normalized per matrix so weights stay comparable across
    frames with different overall similarity levels.
    """
    z = (clip_sim - clip_sim.mean()) / (clip_sim.std() + 1e-8)
    score = weights.clip * z
    if spatial is not None and weights.spatial:
        score = score + weights.spatial * spatial
    if det_conf is not None and weights.det_conf:
        score = score + weights.det_conf * det_conf[None, :]
    return -score


def assignment_components(pairs, clip_sim: np.ndarray,
                          spatial: np.ndarray,
                          det_conf: np.ndarray,
                          cost: np.ndarray,
                          units) -> List[dict]:
    """Raw per-assignment confidence components, for calibration later.

    margin: gap between the chosen cost and the best alternative in the same
    row — the classic ambiguity signal.
    """
    out = []
    for (i, j) in pairs:
        row = cost[i]
        alt = np.delete(row, j)
        margin = float(alt.min() - row[j]) if alt.size else float("inf")
        out.append({
            "unit_idx": int(i),
            "box_idx": int(j),
            "clip_sim": float(clip_sim[i, j]),
            "clip_sim_z": float(
                (clip_sim[i, j] - clip_sim.mean()) / (clip_sim.std() + 1e-8)),
            "spatial": float(spatial[i, j]),
            "det_conf": float(det_conf[j]),
            "margin": margin,
            "fmt_valid": bool(units[i].format_valid),
            "n_units": int(cost.shape[0]),
            "n_boxes": int(cost.shape[1]),
        })
    return out
