"""Hugging Face Hub helpers: pull source frames, push enhanced crops+labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional


def download_frames(repo_id: str, subdir: str, dest: Path,
                    token: Optional[str] = None,
                    shards: Optional[List[str]] = None) -> Path:
    """Snapshot the frames subfolder of a dataset repo to `dest`.

    shards: optional list of shard folder names (e.g. ["shard_000"]) to fetch
            ONLY those subfolders instead of the whole dataset. Great for tuning
            — pull ~1 shard (~1.7GB) instead of all 21 (~35GB). Names without a
            path are matched under subdir; full patterns are used as-is.
    """
    from huggingface_hub import snapshot_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if shards:
        patterns = [s if "/" in s else f"{subdir}/{s}/**" for s in shards]
    else:
        patterns = [f"{subdir}/**"]
    local = snapshot_download(
        repo_id=repo_id, repo_type="dataset", token=token,
        allow_patterns=patterns, local_dir=str(dest),
    )
    return Path(local) / subdir


def list_bucket(bucket_id: str, prefix: Optional[str] = None,
                token: Optional[str] = None):
    """Print + return the files in an HF Storage Bucket (to find your zip's path)."""
    from huggingface_hub import list_bucket_tree
    items = list(list_bucket_tree(bucket_id, prefix=prefix, recursive=True, token=token))
    for it in items:
        name = getattr(it, "path", None) or getattr(it, "name", str(it))
        size = getattr(it, "size", "")
        print(f"  {name}  {size}")
    return items


def download_bucket_zip(bucket_id: str, zip_name: str, dest: Path,
                        token: Optional[str] = None, keep_zip: bool = False) -> Path:
    """Download ONE zip from a bucket (single request) and extract it. Returns the
    directory containing the extracted frames."""
    import zipfile
    from huggingface_hub import download_bucket_files

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    local_zip = dest / Path(zip_name).name
    if not local_zip.exists():
        download_bucket_files(bucket_id, files=[(zip_name, str(local_zip))],
                              token=token, raise_on_missing_files=True)
    if not local_zip.exists():
        raise FileNotFoundError(
            f"'{zip_name}' was not downloaded from bucket '{bucket_id}'. "
            f"Check the exact path with: python scripts/16_run_local.py --bucket {bucket_id} --list")
    extract_dir = dest / "frames"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(extract_dir)
    if not keep_zip:
        local_zip.unlink(missing_ok=True)   # free disk (35GB)
    n = sum(1 for _ in extract_dir.glob("**/*.jpg"))
    print(f"[io] extracted {n} frames from {zip_name} -> {extract_dir}")
    return extract_dir


def hub_file_exists(repo_id: str, path: str, token: Optional[str] = None) -> bool:
    from huggingface_hub import HfApi
    try:
        return HfApi(token=token).file_exists(repo_id, path, repo_type="dataset")
    except Exception:
        return False


def upload_one_file(local: Path, path_in_repo: str, repo_id: str,
                    token: Optional[str] = None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                    repo_id=repo_id, repo_type="dataset")


def make_shard_tar(frames_dir: Path, tar_path: Path) -> int:
    """Tar all .jpg in frames_dir (flat, arcname = filename). Returns file count."""
    import tarfile
    frames_dir, tar_path = Path(frames_dir), Path(tar_path)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(frames_dir.glob("**/*.jpg"))
    with tarfile.open(tar_path, "w") as tf:
        for f in files:
            tf.add(f, arcname=f.name)
    return len(files)


def download_frames_tar(repo_id: str, tar_subdir: str, shard: str, dest: Path,
                        token: Optional[str] = None) -> Path:
    """Download ONE shard tar (1 request) and extract it. Returns the frames dir."""
    import tarfile
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id, f"{tar_subdir}/{shard}.tar", repo_type="dataset", token=token)
    out = Path(dest) / shard
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(p) as t:
        t.extractall(out)
    return out


def download_frame_sample(repo_id: str, subdir: str, dest: Path, n: int = 40,
                          token: Optional[str] = None,
                          shard: Optional[str] = None,
                          seed: Optional[int] = 0) -> Path:
    """Download `n` frames sampled at RANDOM (seconds, not minutes).

    Ideal for tuning: pulls ~n individual files via hf_hub_download instead of the
    whole 5000-file shard.
      shard: a shard name (e.g. "shard_000") to sample within it, or None to sample
             across ALL shards (more variety — recommended for tuning).
      seed:  int for a reproducible random sample; None = take the first n (no shuffle).
    Returns the frames root (dest/subdir) for iter_frames.
    """
    import random
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset")
    prefix = f"{subdir}/{shard}/" if shard else f"{subdir}/"
    pool = [f for f in files
            if f.startswith(prefix) and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if seed is not None:
        random.seed(seed)
        sel = random.sample(pool, min(n, len(pool)))
    else:
        sel = pool[:n]
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    for f in sel:
        hf_hub_download(repo_id, filename=f, repo_type="dataset",
                        token=token, local_dir=str(dest))
    where = shard or "all shards"
    print(f"[io] sampled {len(sel)} frames from {where} (seed={seed}) -> {dest/subdir}")
    return dest / subdir


def iter_frames(frames_dir: Path, pattern: str = "**/*.jpg") -> Iterator[Path]:
    yield from sorted(Path(frames_dir).glob(pattern))


def build_parquet(labels_jsonl: Path, out_parquet: Path) -> Path:
    """Convert the JSONL labels into a single parquet for HF datasets."""
    import pandas as pd

    rows = [json.loads(l) for l in Path(labels_jsonl).open() if l.strip()]
    df = pd.DataFrame(rows)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f"[io] {len(df)} rows -> {out_parquet}")
    return Path(out_parquet)


def build_crops_dataset(crops_dir: Path, labels_jsonl: Path, out_dir: Path,
                        shard_size: int = 5000, move: bool = True):
    """Assemble an HF-ready image dataset: crops sharded into subfolders (<10k each)
    plus a metadata.jsonl that pairs every crop image with its caption + label fields.

    Layout produced under out_dir/crops/:
        shard_000/<crop>.jpg ...          (<= shard_size files each)
        metadata.jsonl                    (HF 'imagefolder' convention:
                                           {"file_name": "shard_000/<crop>.jpg", ...labels})
    With metadata.jsonl, `load_dataset("imagefolder", data_dir=...)` yields each crop
    image together with its caption/fields, and the HF viewer shows them side by side.
    """
    import os
    import shutil

    from . import schema

    crops_dir, labels_jsonl, out_dir = Path(crops_dir), Path(labels_jsonl), Path(out_dir)
    labels = {}
    for line in labels_jsonl.open():
        if line.strip():
            r = json.loads(line)
            labels[Path(r["crop_path"]).name] = r      # key by crop filename (unique)

    crops_out = out_dir / "crops"
    crops_out.mkdir(parents=True, exist_ok=True)
    meta_f = (crops_out / "metadata.jsonl").open("w")
    fields = list(schema.FIELDS) + [schema.MODEL_CONFIDENCE_FIELD, "src_frame", "det_conf"]

    n = 0
    for img in sorted(crops_dir.glob("**/*.jpg")):
        row = labels.get(img.name)
        if row is None:
            continue                                    # skip crops without a label
        shard = "shard_%03d" % (n // shard_size)
        d = crops_out / shard
        d.mkdir(exist_ok=True)
        dest = d / img.name
        if move:
            os.replace(img, dest)
        else:
            shutil.copy(img, dest)
        rec = {"file_name": f"{shard}/{img.name}"}
        for f in fields:
            if f in row:
                rec[f] = row[f]
        meta_f.write(json.dumps(rec) + "\n")
        n += 1
    meta_f.close()
    print(f"[io] packaged {n} crops into {crops_out} "
          f"({(n + shard_size - 1)//shard_size} shards) + metadata.jsonl")
    return crops_out, n


# --------------------------------------------------------------------------- #
# Per-shard streaming (process + upload one shard at a time, resumable)
# --------------------------------------------------------------------------- #
def ensure_repo(repo_id: str, token: Optional[str] = None, private: bool = True):
    from huggingface_hub import HfApi
    HfApi(token=token).create_repo(repo_id, repo_type="dataset",
                                   private=private, exist_ok=True)


def shard_done(repo_id: str, shard: str, token: Optional[str] = None) -> bool:
    """True if _done/<shard> marker exists in the crops repo (shard fully uploaded)."""
    from huggingface_hub import HfApi
    try:
        return HfApi(token=token).file_exists(repo_id, f"_done/{shard}", repo_type="dataset")
    except Exception:
        return False


def mark_shard_done(repo_id: str, shard: str, token: Optional[str] = None):
    import io
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=io.BytesIO(b"done"), path_in_repo=f"_done/{shard}",
        repo_id=repo_id, repo_type="dataset")


def stage_shard(crops_dir: Path, labels_jsonl: Path, stage: Path,
                shard_name: str, shard_size: int = 5000) -> int:
    """Lay out one shard's crops + labels for upload:
        stage/crops/<shard>/part_000/*.jpg   (<=shard_size each)
        stage/labels/<shard>.jsonl           (label rows + repo-relative file_name)
    Returns number of crops staged.
    """
    import shutil
    crops_dir, labels_jsonl, stage = Path(crops_dir), Path(labels_jsonl), Path(stage)
    rows = [json.loads(l) for l in labels_jsonl.open() if l.strip()]
    imgs = {p.name: p for p in crops_dir.glob("**/*.jpg")}
    cdir = stage / "crops" / shard_name
    ldir = stage / "labels"
    cdir.mkdir(parents=True, exist_ok=True)
    ldir.mkdir(parents=True, exist_ok=True)
    out_rows, n = [], 0
    for r in rows:
        name = Path(r["crop_path"]).name
        src = imgs.get(name)
        if src is None:
            continue
        part = "part_%03d" % (n // shard_size)
        d = cdir / part
        d.mkdir(exist_ok=True)
        shutil.copy(src, d / name)
        out_rows.append({**r, "file_name": f"{shard_name}/{part}/{name}"})
        n += 1
    with (ldir / f"{shard_name}.jsonl").open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    return n


def zip_dir(src_dir: Path, zip_path: Path) -> Path:
    """Zip a directory tree (stored, no compression — jpgs are already compressed)."""
    import zipfile
    src_dir, zip_path = Path(src_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for f in src_dir.glob("**/*"):
            if f.is_file():
                z.write(f, arcname=str(f.relative_to(src_dir)))
    return zip_path


def unzip_to(zip_path: Path, dest: Path) -> Path:
    import zipfile
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return dest


def upload_dir(local_dir: Path, repo_id: str, token: Optional[str] = None):
    """Add a staged folder to the crops repo (resumable, skips unchanged files)."""
    from huggingface_hub import HfApi
    HfApi(token=token).upload_large_folder(
        folder_path=str(local_dir), repo_id=repo_id, repo_type="dataset")


def load_local_labels(labels_dir: Path, crops_root: Optional[Path] = None) -> List[dict]:
    """Read all shard label jsonl(s) and tag each row with the crops base dir to use.

    Handles the per-shard-folder layout (D:/LLMSTU/shard_XXX/labels/*.jsonl with a
    sibling D:/LLMSTU/shard_XXX/crops/), the merged layout (labels/ + crops/), and a
    single jsonl + explicit --crops-root. Each returned row gets r["_crops_base"];
    the crop image is then (_crops_base / file_name).
    """
    lp = Path(labels_dir)
    files = [lp] if lp.is_file() else sorted(lp.glob("**/*.jsonl"))
    out: List[dict] = []
    for f in files:
        frows = [json.loads(l) for l in f.open() if l.strip()]
        if not frows:
            continue
        fn0 = frows[0].get("file_name") or Path(frows[0].get("crop_path", "")).name
        candidates = [f.parent.parent / "crops", f.parent / "crops"]
        if crops_root:
            candidates.append(Path(crops_root))
        candidates += [f.parent.parent, f.parent]
        base = next((c for c in candidates if fn0 and (c / fn0).exists()), None)
        base = str(base) if base else (str(crops_root) if crops_root else "")
        for r in frows:
            r["_crops_base"] = base
            out.append(r)
    return out


def list_repo_zips(repo_id: str, token: Optional[str] = None):
    from huggingface_hub import HfApi
    return sorted(f for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
                  if f.endswith(".zip"))


def download_repo_file(repo_id: str, path_in_repo: str, token: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, path_in_repo, repo_type="dataset", token=token)


def delete_repo_file(repo_id: str, path_in_repo: str, token: Optional[str] = None):
    from huggingface_hub import HfApi
    HfApi(token=token).delete_file(path_in_repo=path_in_repo, repo_id=repo_id,
                                   repo_type="dataset")


def gather_labels(repo_id: str, out_jsonl: Path, token: Optional[str] = None) -> int:
    """Download every labels/<shard>.jsonl and concatenate into one file."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    files = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f.startswith("labels/") and f.endswith(".jsonl")]
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w") as out:
        for f in files:
            p = hf_hub_download(repo_id, f, repo_type="dataset", token=token)
            for line in open(p):
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    print(f"[io] gathered {n} labels from {len(files)} shards -> {out_jsonl}")
    return n


def fetch_crops(repo_id: str, file_names, dest: Path, token: Optional[str] = None) -> dict:
    """Download specific crop files (crops/<file_name>) from the repo. Returns {file_name: local_path}."""
    from huggingface_hub import hf_hub_download
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for fn in file_names:
        p = hf_hub_download(repo_id, f"crops/{fn}", repo_type="dataset",
                            token=token, local_dir=str(dest))
        out[fn] = p
    return out


def upload_dataset(local_dir: Path, repo_id: str, token: Optional[str] = None,
                   private: bool = True) -> str:
    """Create (if needed) and upload the enhanced dataset repo (resumable)."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    # upload_large_folder is resumable + parallel — right choice for many crop shards
    api.upload_large_folder(folder_path=str(local_dir), repo_id=repo_id,
                            repo_type="dataset")
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[io] uploaded -> {url}")
    return url
