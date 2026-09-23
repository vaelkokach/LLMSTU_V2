"""The dashboard's per-cue policy, projected onto a regrouped taxonomy.

Two policies in ``tools/dashboard/server.py`` are keyed by the SIX cue classes:
which classes count toward the off-task share, and how long each must persist
before it may page an instructor. A model trained on ``onoff_reliable`` predicts
``on_task``/``off_task``, which appear in neither, so both used to degrade
silently to "nothing is off task, nothing may alert" — a two-class model would
have reported a calm room full of phones.

These tests pin the projection and, more importantly, the two places it is
deliberately ASYMMETRIC: a merged class counts toward an aggregate on weaker
evidence than it is allowed to interrupt a person on.
"""
import pytest

from attention.taxonomy import (CUE_CLASSES, OFF_TASK_CUES, TAXONOMIES,
                                taxonomy_alert_dwell, taxonomy_classes,
                                taxonomy_off_task_classes,
                                taxonomy_off_task_is_impure)

#: The cue-level policy this projects. Mirrors server.ALERT_AFTER_S; if that
#: table changes, this test should be updated deliberately, not silently.
CUE_DWELL = {"phone_use": 15.0, "head_down": 30.0,
             "turned_to_peer": 30.0, "looking_away": 20.0}


def test_cue6_is_the_identity():
    """cue6 must project to exactly the policy as written, or the default
    dashboard behaviour has changed as a side effect of adding a taxonomy."""
    assert taxonomy_off_task_classes("cue6") == list(OFF_TASK_CUES)
    assert taxonomy_alert_dwell("cue6", CUE_DWELL) == CUE_DWELL
    assert taxonomy_off_task_is_impure("cue6") == {}


def test_uncertain_is_neither_on_nor_off_task():
    """An unverifiable crop is not evidence of being off task.

    This is why `uncertain` is in no off-task list and has no dwell: it is the
    one cue that means "we could not see", and counting it would inflate the
    off-task share with occlusion.
    """
    assert "uncertain" not in OFF_TASK_CUES
    assert "uncertain" not in CUE_DWELL
    assert "uncertain" not in taxonomy_off_task_classes("cue6")
    assert "uncertain" not in taxonomy_alert_dwell("cue6", CUE_DWELL)


@pytest.mark.parametrize("name", sorted(TAXONOMIES))
def test_every_class_is_accounted_for(name):
    """Off-task classes are a subset of the taxonomy's own classes, and the
    on-task class is never one of them."""
    classes = taxonomy_classes(name)
    off = taxonomy_off_task_classes(name)
    assert set(off) <= set(classes)
    for cls in off:
        assert "screen_oriented" not in TAXONOMIES[name]["groups"][cls]


@pytest.mark.parametrize("name", sorted(TAXONOMIES))
def test_alertable_classes_are_a_subset_of_off_task_classes(name):
    """Anything allowed to page an instructor must first count as off task.

    The converse is deliberately false — see the next test.
    """
    assert set(taxonomy_alert_dwell(name, CUE_DWELL)) <= \
        set(taxonomy_off_task_classes(name))


def test_a_class_merging_uncertain_counts_but_cannot_alert():
    """The asymmetry, stated as a test.

    ``onoff_reliable``'s ``off_task`` merges head_down + phone_use (actionable)
    with uncertain (not). It contributes to the off-task share, because the
    aggregate is about the class as defined; it may raise NO alert, because an
    alert interrupts a person and a third of the evidence behind it may be a
    student nobody could see.
    """
    assert taxonomy_off_task_classes("onoff_reliable") == ["off_task"]
    assert taxonomy_alert_dwell("onoff_reliable", CUE_DWELL) == {}
    assert "uncertain" in taxonomy_off_task_is_impure("onoff_reliable")["off_task"]


def test_coarse3_can_alert_on_the_class_it_kept_pure():
    """coarse3_reliable keeps phone_use as a singleton, so it stays alertable
    at its own dwell, while down_or_hidden (head_down + uncertain) does not."""
    dwell = taxonomy_alert_dwell("coarse3_reliable", CUE_DWELL)
    assert dwell == {"phone_use": 15.0}
    assert "down_or_hidden" in taxonomy_off_task_classes("coarse3_reliable")
    assert "down_or_hidden" not in dwell


def test_a_merged_class_waits_as_long_as_its_slowest_member():
    """Merging cues makes the evidence weaker, so the wait gets longer.

    Taking the minimum would let a merge BUY a faster alert than either cue
    earned on its own, which is the wrong direction.
    """
    merged = {
        "classes": ["on_task", "slow_merge"],
        "groups": {"on_task": ["screen_oriented"],
                   # 15 s and 30 s; the merge must wait 30, not 15
                   "slow_merge": ["phone_use", "head_down"]},
        "note": "test fixture",
    }
    TAXONOMIES["_test_merge"] = merged
    try:
        assert taxonomy_alert_dwell("_test_merge", CUE_DWELL)["slow_merge"] == 30.0
    finally:
        del TAXONOMIES["_test_merge"]


def test_unknown_taxonomy_raises():
    for fn in (taxonomy_off_task_classes, taxonomy_classes):
        with pytest.raises(KeyError):
            fn("no_such_taxonomy")
    with pytest.raises(KeyError):
        taxonomy_alert_dwell("no_such_taxonomy", CUE_DWELL)


def test_off_task_cues_are_real_cues():
    assert set(OFF_TASK_CUES) <= set(CUE_CLASSES)
    assert set(CUE_DWELL) <= set(CUE_CLASSES)
