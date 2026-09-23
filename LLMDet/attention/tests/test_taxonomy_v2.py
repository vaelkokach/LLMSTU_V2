"""The v2 cue rules, and the guarantee that v1 did not move.

v1 is the rule set every published number and every built sequence was
produced under. If it changes, `llmstu_sequences_*` on disk stop meaning what
the results register says they mean, silently. So the first three tests here
are about v1 NOT changing, and only then about v2 doing what it claims.
"""
import json
from pathlib import Path

import pytest

from attention.taxonomy import (CUE_CLASSES, CUE_TO_ID, RULESETS,
                                _map_record_legacy, candidate_set,
                                cue_conditions, map_record)

REPO = Path(__file__).resolve().parents[3]
SAMPLE = REPO / "grounding_data/llmstu_tools/outputs/gold_candidates.jsonl"
HUMAN = REPO / "event_gold_bundle/gold_annotations_Admin.jsonl"


def _rec(**kw):
    base = {
        "activity": "listening",
        "gaze_direction": "teacher_or_board",
        "attention_target": "instruction",
        "engagement_level": "engaged",
        "posture": "upright",
        "hand_state": "unknown",
        "phone_visible": False,
        "laptop_visible": False,
        "talking": False,
        "occluded": False,
        "face_kpts": 3,
    }
    base.update(kw)
    return base


def _load(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    return [r for r in recs if r.get("status") in (None, "ok")]


# ---------------------------------------------------------------------------
# v1 must not move
# ---------------------------------------------------------------------------

def test_default_ruleset_is_v1():
    """Every existing caller passes no ruleset and must keep getting v1."""
    r = _rec(gaze_direction="down", attention_target="distracted",
             posture="upright", activity="other")
    assert map_record(r) == map_record(r, "v1")
    assert candidate_set(r) == candidate_set(r, "v1")


def test_v1_still_matches_the_legacy_oracle_on_real_records():
    """`_map_record_legacy` is the inlined original, kept as an oracle."""
    recs = _load(SAMPLE)
    mismatched = [r for r in recs if map_record(r, "v1") != _map_record_legacy(r)]
    assert not mismatched, f"{len(mismatched)} records changed under v1"


def test_v1_unchanged_on_human_labels_too():
    recs = _load(HUMAN)
    assert all(map_record(r, "v1") == _map_record_legacy(r) for r in recs)


def test_unknown_ruleset_is_rejected():
    with pytest.raises(KeyError):
        cue_conditions(_rec(), "v3")


def test_both_rulesets_return_the_same_rules_in_the_same_order():
    """map_record and candidate_set both index this list positionally."""
    for rs in RULESETS:
        names = [n for n, _ in cue_conditions(_rec(), rs)]
        assert names == ["uncertain", "phone_use", "head_down",
                         "turned_to_peer", "looking_away", "screen_oriented"]


# ---------------------------------------------------------------------------
# v2: the repair
# ---------------------------------------------------------------------------

def test_looking_down_no_longer_fires_looking_away():
    """The defect, as a single case.

    A student looking down at their own desk, upright, whom the annotator
    called `distracted` because the vocabulary has no better value. v1 fires
    looking_away; v2 reads the gaze field and calls it screen_oriented.
    """
    r = _rec(activity="other", gaze_direction="down",
             attention_target="distracted", posture="upright")
    assert map_record(r, "v1") == CUE_TO_ID["looking_away"]
    assert map_record(r, "v2") == CUE_TO_ID["screen_oriented"]


def test_head_down_no_longer_carries_looking_away_as_a_candidate():
    """This co-occurrence is what collapsed PRODEN's head_down to F1 [value removed]."""
    r = _rec(activity="head_down_sleeping", gaze_direction="down",
             attention_target="distracted", posture="head_down")
    assert map_record(r, "v1") == CUE_TO_ID["head_down"]
    assert map_record(r, "v2") == CUE_TO_ID["head_down"]
    assert CUE_TO_ID["looking_away"] in candidate_set(r, "v1")
    assert CUE_TO_ID["looking_away"] not in candidate_set(r, "v2")


def test_genuine_looking_away_survives():
    """The repair must not empty the class -- only purify it."""
    for r in (_rec(gaze_direction="away_or_window", attention_target="distracted"),
              _rec(activity="looking_away", gaze_direction="unknown",
                   attention_target="distracted")):
        assert map_record(r, "v2") == CUE_TO_ID["looking_away"]


def test_unreadable_crop_becomes_uncertain_not_looking_away():
    """v1 sent no-orientation-signal crops to looking_away via `distracted`."""
    r = _rec(activity="other", gaze_direction="unknown",
             attention_target="distracted", engagement_level="unknown")
    assert map_record(r, "v1") == CUE_TO_ID["looking_away"]
    assert map_record(r, "v2") == CUE_TO_ID["uncertain"]


def test_v2_never_reads_attention_target():
    """The whole claim, asserted directly: perturbing the dropped field
    across every one of its values must not change a single v2 decision."""
    values = ["device", "instruction", "own_work", "distracted", "peer", "unknown"]
    for recs in _load(SAMPLE)[:300], _load(HUMAN)[:300]:
        for r in recs:
            got = {map_record({**r, "attention_target": v}, "v2") for v in values}
            assert len(got) == 1, (r, got)
            cands = {tuple(candidate_set({**r, "attention_target": v}, "v2"))
                     for v in values}
            assert len(cands) == 1


def test_v2_leaves_phone_and_head_down_rules_alone():
    """Only the three rules that read attention_target may change."""
    for recs in (_load(SAMPLE), _load(HUMAN)):
        for r in recs:
            c1, c2 = dict(cue_conditions(r, "v1")), dict(cue_conditions(r, "v2"))
            assert c1["phone_use"] == c2["phone_use"]
            assert c1["head_down"] == c2["head_down"]


def test_v2_candidate_sets_are_never_empty_and_always_contain_the_label():
    """A partial-label loss divides by the candidate count, and the
    single-label loss assumes its target is a member."""
    for recs in (_load(SAMPLE), _load(HUMAN)):
        for r in recs:
            cand = candidate_set(r, "v2")
            assert cand
            assert map_record(r, "v2") in cand
            assert all(0 <= c < len(CUE_CLASSES) for c in cand)


def test_v2_reduces_the_head_down_ambiguity_on_real_records():
    """The measured claim, at corpus scale rather than one handwritten case."""
    recs = _load(SAMPLE)
    la = CUE_TO_ID["looking_away"]
    rates = {}
    for rs in ("v1", "v2"):
        hd = [r for r in recs if map_record(r, rs) == CUE_TO_ID["head_down"]]
        rates[rs] = sum(la in candidate_set(r, rs) for r in hd) / len(hd)
    assert rates["v1"] > 0.95
    assert rates["v2"] < 0.40


def test_v2_purifies_looking_away_on_real_records():
    recs = _load(SAMPLE)
    purity = {}
    for rs in ("v1", "v2"):
        rows = [r for r in recs if map_record(r, rs) == CUE_TO_ID["looking_away"]]
        purity[rs] = sum(r["gaze_direction"] == "away_or_window"
                         for r in rows) / len(rows)
    assert purity["v2"] > purity["v1"] + 0.10


def test_v2_moves_few_hard_labels():
    """The repair is a definition fix, not a relabelling.

    If this ever gets large, the v1/v2 comparison stops being a controlled one
    and the change needs re-justifying rather than quietly re-baselining.
    """
    recs = _load(SAMPLE)
    changed = sum(map_record(r, "v1") != map_record(r, "v2") for r in recs)
    assert changed / len(recs) < 0.10
