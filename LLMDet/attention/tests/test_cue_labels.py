"""The cue-label sidecar: recompute the target without touching the features.

The mechanism's whole value is that a v1-vs-v2 comparison reads the identical
feature file, so these tests are mostly about the ways it could silently read
the WRONG rows -- a misaligned replay, a stale sidecar, a length mismatch --
each of which would produce a plausible number rather than an error.
"""
import json

import numpy as np
import pytest

from attention.taxonomy import CUE_CLASSES, CUE_TO_ID, map_record
from attention.thesis_eval import data as D
from attention.thesis_eval.build_cue_labels import build

VIDEO = "video_0001_0_10_x_y"
N_FRAMES = 12


def _rec(i, **kw):
    """One label record at second ``i``, on a stationary student."""
    stem = f"t{i:06d}_000_f{i * 20:06d}_{VIDEO}"
    base = {
        "src_frame": f"{stem}.jpg",
        "file_name": f"shard_000/{stem}__p00.jpg",
        "bbox_person": [400.0, 300.0, 600.0, 700.0],
        "head_span_px": 150.0,
        "activity": "using_laptop",
        "gaze_direction": "laptop",
        "attention_target": "device",
        "engagement_level": "engaged",
        "posture": "upright",
        "hand_state": "unknown",
        "phone_visible": False,
        "laptop_visible": True,
        "talking": False,
        "occluded": False,
        "face_kpts": 3,
    }
    base.update(kw)
    return base


@pytest.fixture
def corpus(tmp_path):
    """A one-video, one-seat corpus plus the sequence npz the builder would
    have written for it, with features that are deliberately recognisable."""
    # Frames 4..7 are the defect: looking down, called `distracted`, upright
    # posture. v1 -> looking_away, v2 -> screen_oriented.
    recs = []
    for i in range(N_FRAMES):
        if 4 <= i < 8:
            recs.append(_rec(i, activity="other", gaze_direction="down",
                             attention_target="distracted", posture="upright"))
        else:
            recs.append(_rec(i))

    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    f2v = tmp_path / "frame_to_video.json"
    f2v.write_text(json.dumps({r["src_frame"]: VIDEO for r in recs}), encoding="utf-8")

    # split_videos(seed=42, val_fraction=0.2) on a single video puts it in
    # train, so the builder would have written train/sample_000000.npz.
    root = tmp_path / "seq"
    (root / "train").mkdir(parents=True)
    key = "train/sample_000000.npz"
    x = np.arange(N_FRAMES * 4, dtype=np.float32).reshape(N_FRAMES, 4)
    y = np.array([map_record(r, "v1") for r in recs], dtype=np.int64)
    t = np.array([float(i) for i in range(N_FRAMES)], dtype=np.float64)
    cand = np.zeros((N_FRAMES, len(CUE_CLASSES)), dtype=np.uint8)
    cand[np.arange(N_FRAMES), y] = 1
    # `layout` is declared so the loader validates rather than assuming v570;
    # the test registers a matching 4-column layout below.
    np.savez_compressed(root / key, x=x, y_frames=y, y=int(y[0]), t=t,
                        y_cand=cand, layout="_test4")

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": [
        {"file": key, "video_id": VIDEO, "seat_id": 0, "split": "train"}]}),
        encoding="utf-8")
    return {"root": root, "labels": labels, "f2v": f2v, "key": key,
            "manifest": manifest, "x": x, "y_v1": y, "t": t, "recs": recs}


# ---------------------------------------------------------------------------
# the builder's own proofs
# ---------------------------------------------------------------------------

def test_v1_rebuild_reproduces_the_stored_labels_exactly(corpus):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v1")
    assert list(out["keys"]) == [corpus["key"]]
    np.testing.assert_array_equal(out["y"], corpus["y_v1"])
    assert out["n_frames_changed"] == 0


def test_v2_changes_only_the_defective_frames(corpus):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")
    y2 = out["y"]
    moved = np.nonzero(y2 != corpus["y_v1"])[0]
    np.testing.assert_array_equal(moved, np.arange(4, 8))
    assert set(corpus["y_v1"][4:8]) == {CUE_TO_ID["looking_away"]}
    assert set(y2[4:8]) == {CUE_TO_ID["screen_oriented"]}


def test_v2_drops_looking_away_from_the_candidate_sets(corpus):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")
    assert not out["y_cand"][4:8, CUE_TO_ID["looking_away"]].any()
    assert out["y_cand"].sum(axis=1).min() >= 1


def test_a_drifted_replay_aborts_rather_than_patching_wrong_rows(corpus):
    """Timestamps are the proof that the replay recovered the same frames."""
    z = dict(np.load(corpus["root"] / corpus["key"]))
    z["t"] = z["t"] + 1.0
    np.savez_compressed(corpus["root"] / corpus["key"], **z)
    with pytest.raises(SystemExit, match="timestamp mismatch"):
        build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")


def test_changed_rules_abort_the_v1_self_check(corpus):
    """If the stored labels disagree with a v1 recompute, the build must stop:
    a v2 set written on top would be wrong for an unrelated reason."""
    z = dict(np.load(corpus["root"] / corpus["key"]))
    z["y_frames"] = np.zeros_like(z["y_frames"])
    np.savez_compressed(corpus["root"] / corpus["key"], **z)
    with pytest.raises(SystemExit, match="differ from the stored labels"):
        build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")


def test_a_manifest_entry_with_no_recomputed_label_aborts(corpus):
    extra = json.loads(corpus["manifest"].read_text())
    extra["samples"].append({"file": "train/sample_000099.npz",
                             "video_id": VIDEO, "seat_id": 1, "split": "train"})
    corpus["manifest"].write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(SystemExit, match="no recomputed"):
        build(corpus["root"], corpus["labels"], corpus["f2v"], "v2",
              corpus["manifest"])


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------

def _write_sidecar(tmp_path, out, name="cue.npz"):
    p = tmp_path / name
    np.savez_compressed(p, keys=out["keys"].astype(str), offsets=out["offsets"],
                        y=out["y"], y_cand=out["y_cand"], ruleset=out["ruleset"])
    return p


def test_load_split_swaps_the_target_and_leaves_features_alone(corpus, tmp_path):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")
    side = _write_sidecar(tmp_path, out)
    D.FEATURE_CONFIGS["_test4"] = ["_t"]
    D.LAYOUTS["_test4"] = {"_t": (0, 4)}
    D.CONFIG_LAYOUT["_test4"] = "_test4"
    D.LAYOUT_WIDTH["_test4"] = 4
    try:
        base = D.load_split(corpus["manifest"], corpus["root"], "train", "_test4")
        over = D.load_split(corpus["manifest"], corpus["root"], "train", "_test4",
                            cue_labels=side)
        np.testing.assert_array_equal(base[0].x, over[0].x)   # features untouched
        np.testing.assert_array_equal(base[0].y, corpus["y_v1"])
        np.testing.assert_array_equal(over[0].y, out["y"])
        assert not np.array_equal(base[0].y, over[0].y)
    finally:
        for d in (D.FEATURE_CONFIGS, D.LAYOUTS, D.CONFIG_LAYOUT, D.LAYOUT_WIDTH):
            d.pop("_test4", None)


def test_a_sidecar_missing_a_sequence_is_fatal(corpus, tmp_path):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")
    out["keys"] = np.array(["train/sample_999999.npz"], dtype=object)
    side = _write_sidecar(tmp_path, out, "stale.npz")
    cl = D.CueLabels(side)
    with pytest.raises(RuntimeError, match="no entry"):
        cl.get(corpus["key"], N_FRAMES)


def test_a_length_mismatch_is_fatal(corpus, tmp_path):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")
    side = _write_sidecar(tmp_path, out)
    cl = D.CueLabels(side)
    with pytest.raises(RuntimeError, match="different frames"):
        cl.get(corpus["key"], N_FRAMES + 1)


def test_ruleset_travels_with_the_sidecar(corpus, tmp_path):
    out = build(corpus["root"], corpus["labels"], corpus["f2v"], "v2")
    assert D.CueLabels(_write_sidecar(tmp_path, out)).ruleset == "v2"
