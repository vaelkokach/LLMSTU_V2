"""Near-duplicate crop removal for 1-fps footage.

At 1 frame/second a student who sits still yields many near-identical crops. This
collapses those runs: crops are grouped by (video, seat region), ordered in time,
and a crop is kept only if it differs enough (perceptual dHash Hamming distance)
from the last KEPT crop in that group. Slow drift still produces new keyframes when
the scene actually changes, so we don't lose real activity — only redundancy.

Run it BETWEEN cropping and captioning to cut the caption bill dramatically:
    crop -> dedup (this) -> caption on the deduped manifest
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image

_VIDEO_RE = re.compile(r"video_\d+", re.IGNORECASE)


def dhash(path: Path, size: int = 8) -> int:
    """Row-wise difference hash -> integer of size*size bits."""
    import numpy as np
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _video_key(src_frame: str) -> str:
    m = _VIDEO_RE.search(src_frame or "")
    return m.group(0) if m else (src_frame or "")


def _cell(bbox, cell_px: int):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return (int(cx // cell_px), int(cy // cell_px))


def _resolve(crops_dir: Path, row) -> Optional[Path]:
    p = crops_dir / Path(row["crop_path"]).name
    if p.exists():
        return p
    p = crops_dir.parent / row["crop_path"]
    return p if p.exists() else None


def dedup_manifest(manifest_path: Path, crops_dir: Path, out_manifest: Path,
                   hamming_threshold: int = 6, cell_px: int = 64,
                   hash_size: int = 8) -> dict:
    """Write a deduped copy of the crop manifest. Returns {kept, dropped, total}."""
    manifest_path, crops_dir, out_manifest = (
        Path(manifest_path), Path(crops_dir), Path(out_manifest))
    rows = [json.loads(l) for l in manifest_path.open() if l.strip()]

    groups = defaultdict(list)
    for r in rows:
        key = (_video_key(r.get("src_frame", "")),
               _cell(r.get("bbox_crop", [0, 0, 0, 0]), cell_px))
        groups[key].append(r)

    kept = []
    for _, items in groups.items():
        items.sort(key=lambda r: r.get("src_frame", ""))   # temporal order
        last_hash = None
        for r in items:
            p = _resolve(crops_dir, r)
            if p is None:
                kept.append(r)                               # can't hash -> keep
                continue
            try:
                h = dhash(p, hash_size)
            except Exception:
                kept.append(r)
                continue
            if last_hash is None or hamming(h, last_hash) >= hamming_threshold:
                kept.append(r)
                last_hash = h
            # else: near-duplicate of the last kept crop in this seat -> drop

    with out_manifest.open("w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    stats = {"total": len(rows), "kept": len(kept), "dropped": len(rows) - len(kept)}
    pct = 100 * stats["dropped"] / max(1, stats["total"])
    print(f"[dedup] {stats['kept']}/{stats['total']} kept "
          f"({stats['dropped']} dropped, -{pct:.1f}%) -> {out_manifest}")
    return stats
