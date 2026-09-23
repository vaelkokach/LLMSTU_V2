from attention.taxonomy import CUE_CLASSES, CUE_TO_ID, map_record, parse_stem_time


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


def test_class_set():
    assert len(CUE_CLASSES) == 6
    assert CUE_CLASSES[0] == "screen_oriented"


def test_screen_oriented_listening():
    assert map_record(_rec()) == CUE_TO_ID["screen_oriented"]


def test_screen_oriented_laptop():
    r = _rec(activity="using_laptop", gaze_direction="laptop", attention_target="device")
    assert map_record(r) == CUE_TO_ID["screen_oriented"]


def test_phone_beats_laptop():
    r = _rec(activity="using_laptop", phone_visible=True)
    assert map_record(r) == CUE_TO_ID["phone_use"]


def test_phone_from_hand_state():
    assert map_record(_rec(hand_state="on_phone")) == CUE_TO_ID["phone_use"]


def test_head_down_activity():
    r = _rec(activity="head_down_sleeping", posture="leaning_back")
    assert map_record(r) == CUE_TO_ID["head_down"]


def test_head_down_posture_slumped():
    assert map_record(_rec(posture="slumped")) == CUE_TO_ID["head_down"]


def test_turned_to_peer():
    assert map_record(_rec(activity="talking_to_peer")) == CUE_TO_ID["turned_to_peer"]
    assert map_record(_rec(gaze_direction="peer", attention_target="peer")) == CUE_TO_ID["turned_to_peer"]


def test_looking_away():
    r = _rec(activity="looking_away", gaze_direction="away_or_window", attention_target="distracted")
    assert map_record(r) == CUE_TO_ID["looking_away"]


def test_uncertain_occluded_no_face():
    r = _rec(occluded=True, face_kpts=2)
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_occluded_with_face_still_labeled():
    r = _rec(occluded=True, face_kpts=3, activity="using_laptop")
    assert map_record(r) == CUE_TO_ID["screen_oriented"]


def test_uncertain_all_unknown():
    r = _rec(activity="other", gaze_direction="unknown", attention_target="unknown", engagement_level="unknown")
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_fallback_eating_maps_to_uncertain():
    # idle_other merged into uncertain (fallback fired on only 16/283k records)
    r = _rec(activity="eating_drinking", gaze_direction="unknown", attention_target="unknown")
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_uncertain_outranks_phone():
    r = _rec(occluded=True, face_kpts=2, phone_visible=True)
    assert map_record(r) == CUE_TO_ID["uncertain"]


def test_parse_stem_time():
    assert parse_stem_time("t000033_856_f000680") == 33.856
    assert parse_stem_time("t000000_000_f000000_video_0005_0_10_x_y") == 0.0
    assert parse_stem_time("garbage") is None


# ---------------------------------------------------------------------------
# Coarser taxonomies (TAXONOMIES / taxonomy_lut)
# ---------------------------------------------------------------------------

def test_cue6_taxonomy_is_the_identity():
    """The default must not change behaviour: it is applied to every load."""
    from attention.taxonomy import taxonomy_lut, taxonomy_classes, CUE_CLASSES
    assert taxonomy_lut("cue6") == list(range(len(CUE_CLASSES)))
    assert taxonomy_classes("cue6") == list(CUE_CLASSES)


def test_every_source_class_is_mapped_or_deliberately_excluded():
    """No class may fall through silently.

    A source class that is neither grouped nor listed as excluded would be
    relabelled to IGNORE by accident, quietly shrinking the evaluation set and
    improving the score for a reason nobody chose.
    """
    from attention.taxonomy import (IGNORE_LABEL, TAXONOMIES,
                                    label_space_classes, taxonomy_lut,
                                    taxonomy_excluded, taxonomy_space)
    for name, spec in TAXONOMIES.items():
        lut = taxonomy_lut(name)
        # A taxonomy is defined over ONE label space and must account for every
        # class of THAT space -- cue6 for the regroupings, cue9 for the split.
        src = label_space_classes(taxonomy_space(name))
        assert len(lut) == len(src), name
        grouped = {c for g in spec["groups"].values() for c in g}
        excluded = set(taxonomy_excluded(name))
        assert grouped | excluded == set(src), name
        assert not (grouped & excluded), f"{name}: class both grouped and excluded"
        for new_id in lut:
            assert new_id == IGNORE_LABEL or 0 <= new_id < len(spec["classes"]), name


def test_reliable_taxonomies_abstain_on_the_unsupported_classes():
    """The two classes the labels cannot support must be the excluded ones."""
    from attention.taxonomy import taxonomy_excluded
    for name in ("onoff_reliable", "coarse3_reliable"):
        assert set(taxonomy_excluded(name)) == {"looking_away", "turned_to_peer"}, name
    # ...and the all-frames variants must abstain on nothing.
    for name in ("cue6", "onoff"):
        assert taxonomy_excluded(name) == [], name


def test_group_members_land_on_the_same_new_id():
    from attention.taxonomy import (TAXONOMIES, label_space_classes,
                                    taxonomy_lut, taxonomy_space)
    for name, spec in TAXONOMIES.items():
        lut = taxonomy_lut(name)
        src_to_id = {c: i for i, c in
                     enumerate(label_space_classes(taxonomy_space(name)))}
        for gname, members in spec["groups"].items():
            ids = {lut[src_to_id[m]] for m in members}
            assert len(ids) == 1, f"{name}/{gname} split across ids {ids}"


def test_a_taxonomy_cannot_group_a_class_outside_its_space():
    """cue6's `screen_oriented` is not a cue9 class and vice versa.

    Both spaces are just integer ids, so grouping a name from the wrong one
    would either raise here or -- worse -- land on an id that exists and mean a
    different class. taxonomy_lut refuses it by name.
    """
    import pytest
    from attention.taxonomy import TAXONOMIES, taxonomy_lut
    TAXONOMIES["_test_bad_space"] = {
        "space": "cue9",
        "classes": ["x"],
        # screen_oriented exists in cue6, not in cue9
        "groups": {"x": ["screen_oriented"]},
        "note": "test fixture",
    }
    try:
        with pytest.raises(KeyError, match="not a class of its label space"):
            taxonomy_lut("_test_bad_space")
    finally:
        del TAXONOMIES["_test_bad_space"]


def test_cue9_splits_screen_oriented_and_keeps_the_rest():
    """The shape of the split, pinned.

    cue9 replaces `screen_oriented` with four classes and keeps the other five
    cue6 names, so the off-task side of the taxonomy is unchanged and every
    number about `phone_use` or `head_down` stays about the same thing.
    """
    from attention.taxonomy import CUE9_CLASSES, CUE9_ON_TASK, CUE_CLASSES
    assert len(CUE9_CLASSES) == 9
    assert "screen_oriented" not in CUE9_CLASSES
    assert set(CUE9_ON_TASK) == {"writing_notes", "using_laptop", "reading",
                                 "listening"}
    assert set(CUE9_ON_TASK) <= set(CUE9_CLASSES)
    kept = set(CUE_CLASSES) - {"screen_oriented"}
    assert kept <= set(CUE9_CLASSES), "cue9 must keep the five non-on-task cues"
    assert set(CUE9_CLASSES) == kept | set(CUE9_ON_TASK)


def test_cue9_sends_gaze_down_to_head_down_not_looking_away():
    """The one changed rule, and the reason it is the right change.

    RULESET_V2_RATIONALE: `gaze == down` used to reach `looking_away` (via
    `target == distracted`), so *looking down* fired *looking away*. In cue9 it
    fires `head_down` directly, and `head_down` outranks `looking_away`.
    """
    from attention.taxonomy import (CUE9_CLASSES, CUE_CLASSES, map_record,
                                    map_record_cue9)
    rec = {"activity": "other", "gaze_direction": "down",
           "attention_target": "distracted", "engagement_level": "engaged",
           "posture": "upright", "hand_state": "on_desk_idle",
           "occluded": False, "face_kpts": 3, "phone_visible": False,
           "talking": False}
    assert CUE_CLASSES[map_record(rec)] == "looking_away"
    assert CUE9_CLASSES[map_record_cue9(rec)] == "head_down"


def test_cue9_label_is_always_in_its_own_candidate_set():
    """Same invariant cue6 has: the precedence winner must be a candidate, or a
    partial-label objective would be scored against a set excluding the target."""
    from attention.taxonomy import candidate_set_cue9, map_record_cue9
    import itertools
    fields = {
        "activity": ["writing_notes", "using_laptop", "reading", "listening",
                     "using_phone", "head_down_sleeping", "talking_to_peer",
                     "looking_away", "other"],
        "gaze_direction": ["laptop", "own_desk", "teacher_or_board", "down",
                           "away_or_window", "phone", "peer", "unknown"],
        "attention_target": ["device", "instruction", "own_work", "distracted",
                             "peer", "unknown"],
    }
    keys = list(fields)
    for combo in itertools.product(*(fields[k] for k in keys)):
        rec = dict(zip(keys, combo))
        rec.update({"posture": "upright", "hand_state": "on_desk_idle",
                    "engagement_level": "engaged", "occluded": False,
                    "face_kpts": 3, "phone_visible": False, "talking": False})
        assert map_record_cue9(rec) in candidate_set_cue9(rec), rec
