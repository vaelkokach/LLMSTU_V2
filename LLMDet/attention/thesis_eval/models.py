"""Temporal architectures evaluated on the project's per-frame cue sequences.

All models share one contract so the evaluator never needs to know which is
loaded:

    forward(x[B, T, D], pad_mask[B, T] bool True-at-padding)
        -> {"logits": [B, T, C], optional "aux_logits": [S, B, T, C],
            optional "boundary": [B, T]}

``transformer``  the project's existing 4-layer encoder (reference).
``mstcn``        MS-TCN / MS-TCN++ style multi-stage dilated TCN — the standard
                 action-segmentation baseline the addendum asks for first.
``asrf``         MS-TCN backbone with a parallel boundary-regression branch and
                 boundary-driven refinement of the frame labels. This is the
                 architecture that directly targets the project's measured
                 failure: frame accuracy reaches 90% of the teacher ceiling
                 while event recall reaches only ~50% of it, i.e. the errors sit
                 at episode boundaries [internal notes, not included].

Padding is handled explicitly everywhere: the TCN variants multiply by the
validity mask after every layer, because a zero-padded tail otherwise leaks
into the receptive field of real frames through the dilated convolutions.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention.temporal_model import AttentionTransformer


class TransformerWrapper(nn.Module):
    """Adapts the existing :class:`AttentionTransformer` to the shared contract.

    ``per_frame`` is forced True: its default is False and the checkpoint loads
    cleanly either way, which already produced one silently flat timeline
    [internal notes, not included] bug 1).
    """

    def __init__(self, input_dim: int, num_classes: int = 6, hidden_dim: int = 512,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1,
                 max_seq_len: int = 128):
        super().__init__()
        self.net = AttentionTransformer(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=num_heads, dropout=dropout, num_classes=num_classes,
            max_seq_len=max_seq_len, per_frame=True)

    def forward(self, x, pad_mask=None) -> Dict[str, torch.Tensor]:
        return {"logits": self.net(x, key_padding_mask=pad_mask)}


class _DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv1x1 = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid):
        # valid: [B, 1, T] float, 1 at real frames. Re-masking after every
        # layer stops the zero padding from propagating into real receptive
        # fields through the dilations.
        out = F.relu(self.conv(x * valid))
        out = self.drop(self.conv1x1(out))
        return (x + out) * valid


class _SingleStageTCN(nn.Module):
    def __init__(self, in_dim: int, channels: int, num_layers: int,
                 out_dim: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, 1)
        self.layers = nn.ModuleList(
            [_DilatedResidualLayer(2 ** i, channels, dropout) for i in range(num_layers)])
        self.out = nn.Conv1d(channels, out_dim, 1)

    def forward(self, x, valid):
        z = self.inp(x) * valid
        for layer in self.layers:
            z = layer(z, valid)
        return self.out(z) * valid, z


class MSTCN(nn.Module):
    """Multi-stage temporal convolutional network (Farha & Gall).

    Each refinement stage consumes the previous stage's softmax and re-predicts,
    which is what suppresses the over-segmentation a single stage produces.
    Every stage's logits are returned so the smoothing loss can be applied to
    all of them, as in the reference implementation.
    """

    def __init__(self, input_dim: int, num_classes: int = 6, channels: int = 128,
                 num_layers: int = 10, num_stages: int = 4, dropout: float = 0.5):
        super().__init__()
        self.stage1 = _SingleStageTCN(input_dim, channels, num_layers, num_classes, dropout)
        self.stages = nn.ModuleList(
            [_SingleStageTCN(num_classes, channels, num_layers, num_classes, dropout)
             for _ in range(num_stages - 1)])

    def forward(self, x, pad_mask=None) -> Dict[str, torch.Tensor]:
        xt = x.transpose(1, 2)                                   # [B, D, T]
        valid = (torch.ones_like(xt[:, :1]) if pad_mask is None
                 else (~pad_mask).unsqueeze(1).to(xt.dtype))     # [B, 1, T]
        out, _ = self.stage1(xt, valid)
        outs: List[torch.Tensor] = [out]
        for st in self.stages:
            out, _ = st(F.softmax(out, dim=1) * valid, valid)
            outs.append(out)
        stacked = torch.stack([o.transpose(1, 2) for o in outs], dim=0)  # [S, B, T, C]
        return {"logits": stacked[-1], "aux_logits": stacked}


class ASRF(nn.Module):
    """Action Segment Refinement Framework (Ishikawa et al.).

    A shared dilated-TCN encoder feeds two heads:

    * an **action** head predicting the frame class, and
    * a **boundary** head regressing the probability that a frame is a cue
      transition.

    At inference the boundary probabilities cut the timeline into segments and
    each segment is relabelled by its mean class posterior. This converts a
    fragmented per-frame prediction into clean episodes without touching the
    taxonomy — the project's event layer currently has to recover boundaries
    from noisy argmax runs with only hysteresis to help it.
    """

    def __init__(self, input_dim: int, num_classes: int = 6, channels: int = 128,
                 num_layers: int = 10, num_stages: int = 4, dropout: float = 0.5,
                 boundary_threshold: float = 0.5):
        super().__init__()
        self.boundary_threshold = boundary_threshold
        self.backbone = _SingleStageTCN(input_dim, channels, num_layers, num_classes, dropout)
        self.cls_stages = nn.ModuleList(
            [_SingleStageTCN(num_classes, channels, num_layers, num_classes, dropout)
             for _ in range(num_stages - 1)])
        self.bnd_stages = nn.ModuleList(
            [_SingleStageTCN(1, channels, num_layers, 1, dropout)
             for _ in range(num_stages - 1)])
        self.bnd_head = nn.Conv1d(channels, 1, 1)

    def forward(self, x, pad_mask=None) -> Dict[str, torch.Tensor]:
        xt = x.transpose(1, 2)
        valid = (torch.ones_like(xt[:, :1]) if pad_mask is None
                 else (~pad_mask).unsqueeze(1).to(xt.dtype))
        cls_out, feat = self.backbone(xt, valid)
        bnd_out = self.bnd_head(feat) * valid
        cls_outs, bnd_outs = [cls_out], [bnd_out]
        for cs, bs in zip(self.cls_stages, self.bnd_stages):
            cls_out, _ = cs(F.softmax(cls_out, dim=1) * valid, valid)
            bnd_out, _ = bs(torch.sigmoid(bnd_out) * valid, valid)
            cls_outs.append(cls_out)
            bnd_outs.append(bnd_out)
        return {
            "logits": cls_outs[-1].transpose(1, 2),
            "aux_logits": torch.stack([o.transpose(1, 2) for o in cls_outs], dim=0),
            "boundary": bnd_outs[-1].squeeze(1),                      # [B, T] logits
            "aux_boundary": torch.stack([o.squeeze(1) for o in bnd_outs], dim=0),
        }

    @torch.no_grad()
    def refine(self, logits: torch.Tensor, boundary_logits: torch.Tensor,
               lengths: torch.Tensor) -> torch.Tensor:
        """Boundary-driven relabelling. Returns [B, T] refined class ids.

        A frame starts a new segment when its boundary probability exceeds the
        threshold; every frame in a segment takes the argmax of the segment's
        *mean* posterior. Averaging over the segment is what removes the
        single-frame flicker that ordinary argmax leaves behind.
        """
        probs = torch.softmax(logits, dim=-1)
        bprob = torch.sigmoid(boundary_logits)
        out = probs.argmax(-1).clone()
        for b in range(probs.size(0)):
            T = int(lengths[b])
            if T == 0:
                continue
            cuts = [0] + [i for i in range(1, T)
                          if bprob[b, i] > self.boundary_threshold] + [T]
            for s, e in zip(cuts[:-1], cuts[1:]):
                if e > s:
                    out[b, s:e] = probs[b, s:e].mean(0).argmax()
        return out


def boundary_targets_from_labels(y: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """1.0 where the cue changes between t-1 and t, else 0.0; 0 at padding.

    Frame 0 is *not* marked: it is a sequence start, not an observed transition,
    and marking it would teach the model that every track opens with a boundary.
    """
    b = torch.zeros_like(y, dtype=torch.float32)
    valid = y != ignore_index
    changed = (y[:, 1:] != y[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    b[:, 1:] = changed.to(torch.float32)
    return b * valid.to(torch.float32)


def build_model(name: str, input_dim: int, num_classes: int = 6, **kw) -> nn.Module:
    name = name.lower()
    if name == "transformer":
        return TransformerWrapper(input_dim, num_classes, **kw)
    if name in ("mstcn", "ms-tcn"):
        return MSTCN(input_dim, num_classes, **kw)
    if name == "asrf":
        return ASRF(input_dim, num_classes, **kw)
    raise KeyError(f"unknown model {name!r}")
