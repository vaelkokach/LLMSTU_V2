"""Leakage and integrity tests for the frozen Branch-C fold manifest.

The manifest is the one artifact the whole branch's validity rests on. These
tests are the mechanical part of that guarantee: they prove no video crosses a
fold boundary, that the manifest matches the corpus it claims to partition, and
that its recorded hash still describes its contents.

They must run without a GPU and without loading a single feature array.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "outputs/branch_c/splits/branch_c_folds.json"
SEQ_META = REPO / "grounding_data/llmstu_sequences_full/meta.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="Branch-C fold manifest not built"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_hash_matches_contents(manifest):
    """The recorded sha256 must still describe the file. Catches silent edits."""
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    payload = json.dumps(body, indent=2, sort_keys=True)
    assert hashlib.sha256(payload.encode()).hexdigest() == manifest["manifest_sha256"], (
        "fold manifest content does not match its recorded hash; it has been "
        "edited after freezing"
    )


def test_no_video_crosses_an_outer_fold(manifest):
    """Every video appears in exactly one outer-test fold."""
    seen: dict[str, int] = {}
    for fold in manifest["folds"]:
        for v in fold["outer_test_videos"]:
            assert v not in seen, (
                f"{v} is in outer-test of fold {seen[v]} and fold {fold['fold']}"
            )
            seen[v] = fold["fold"]
    assert len(seen) == manifest["n_videos"]


def test_outer_test_disjoint_from_its_own_training_data(manifest):
    """The decisive test: no fold may train on a video it is scored on."""
    for fold in manifest["folds"]:
        test = set(fold["outer_test_videos"])
        train = set(fold["inner_train_videos"])
        val = set(fold["inner_val_videos"])
        assert not (test & train), f"fold {fold['fold']}: outer test leaks into inner train"
        assert not (test & val), f"fold {fold['fold']}: outer test leaks into inner val"


def test_inner_train_and_inner_val_are_disjoint(manifest):
    for fold in manifest["folds"]:
        assert not (set(fold["inner_train_videos"]) & set(fold["inner_val_videos"])), (
            f"fold {fold['fold']}: inner train and inner val overlap"
        )


def test_each_fold_partitions_the_whole_corpus(manifest):
    """test + inner_train + inner_val must cover every video exactly once."""
    n = manifest["n_videos"]
    for fold in manifest["folds"]:
        parts = (
            fold["outer_test_videos"]
            + fold["inner_train_videos"]
            + fold["inner_val_videos"]
        )
        assert len(parts) == n, f"fold {fold['fold']} covers {len(parts)} of {n} videos"
        assert len(set(parts)) == n, f"fold {fold['fold']} lists a video twice"


def test_manifest_matches_the_corpus_it_claims_to_partition(manifest):
    """Guards against the manifest drifting from a regenerated sequence set."""
    if not SEQ_META.exists():
        pytest.skip("sequence meta.json unavailable")
    meta = json.loads(SEQ_META.read_text())
    corpus = {s["video_id"] for s in meta["samples"]}
    listed = set()
    for fold in manifest["folds"]:
        listed |= set(fold["outer_test_videos"])
    assert listed == corpus, (
        "fold manifest and sequence corpus disagree; "
        f"only-in-manifest={sorted(listed - corpus)[:5]} "
        f"only-in-corpus={sorted(corpus - listed)[:5]}"
    )
    assert manifest["n_sequences"] == len(meta["samples"])


def test_class_support_is_balanced_across_folds(manifest):
    """A fold with near-zero support for a class makes its per-class F1 undefined.

    Not a leakage property, but a validity one: the protocol commits to reporting
    per-class F1 over pooled outer predictions, and a wildly unbalanced fold would
    make the per-fold diagnostics that sit beside it meaningless.
    """
    supports = [f["outer_test_class_sequences"] for f in manifest["folds"]]
    n_classes = len(manifest["cue_classes"])
    for c in range(n_classes):
        col = [s[c] for s in supports]
        assert min(col) > 0, (
            f"class {manifest['cue_classes'][c]} has zero support in some fold: {col}"
        )
        # every fold within 2x of the smallest for that class
        assert max(col) <= 2 * min(col), (
            f"class {manifest['cue_classes'][c]} support varies more than 2x "
            f"across folds: {col}"
        )


def test_grouping_unit_is_recorded(manifest):
    """The protocol's honesty depends on the grouping caveat travelling with it."""
    assert manifest["grouping_unit"] == "video_id"
    assert "one fixed camera pose" in manifest["grouping_note"]
    assert "not identified across" in manifest["grouping_note"]
