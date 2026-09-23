"""Visible-cue taxonomy for the classroom attention pipeline.

Maps the structured LLMSTU per-student labels (activity, gaze_direction,
attention_target, posture, hand_state, booleans) onto a closed set of
observable cue classes. The taxonomy deliberately describes *visible
behavior*, not inferred mental state: we claim "the student's head is down",
never "the student is not paying attention".

Cue classes (precedence order used by :func:`map_record`, first match wins):

===================  ==========================================================
cue                  fires when
===================  ==========================================================
uncertain            occluded AND face_kpts <= UNCERTAIN_FACE_KPTS, or the
                     record carries no usable orientation signal
                     (gaze/attention/engagement/activity all unknown-ish)
phone_use            activity == using_phone, phone_visible == True,
                     gaze_direction == phone, or hand_state == on_phone
head_down            activity == head_down_sleeping, or posture in
                     {head_down, slumped}
turned_to_peer       activity == talking_to_peer, talking == True,
                     gaze_direction == peer, attention_target == peer
looking_away         gaze_direction == away_or_window, activity ==
                     looking_away, or attention_target == distracted
screen_oriented      task-oriented orientation: activity in {using_laptop,
                     listening, reading, writing_notes} or gaze_direction in
                     {laptop, teacher_or_board, own_desk, down} with a
                     task-consistent attention_target. NOTE: this class
                     covers both screen AND instructor/board orientation —
                     in a computer-lab lecture both are on-task.
uncertain (fallback) everything else (no orientation cue and no specific
                     activity match). Measured on LLMSTU this fallback fires
                     on only 16/283k records — a separate idle_other class is
                     not supported by the data, and the annotation guideline
                     defines exactly this situation as "uncertain".
===================  ==========================================================

The precedence encodes "specific off-task cue beats generic on-task cue":
a student who is using_laptop but has a visible phone is phone_use.
`uncertain` outranks everything because labels on heavily occluded students
are not verifiable from pixels (see annotation guideline).
"""

from typing import Dict, List, Optional, Tuple

CUE_CLASSES: List[str] = [
    "screen_oriented",
    "looking_away",
    "head_down",
    "turned_to_peer",
    "phone_use",
    "uncertain",
]
CUE_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CUE_CLASSES)}
NUM_CUE_CLASSES: int = len(CUE_CLASSES)

# NPZ sequences built with the pre-2026-07-29 7-class taxonomy store
# idle_other as id 5 and uncertain as id 6; both fold into uncertain (5).
LEGACY_7CLASS_REMAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}

# On-task score per cue, used for the aggregate classroom overlay only
# (NOT a claim about individual mental state).
CUE_TASK_SCORE: Dict[str, float] = {
    "screen_oriented": 1.0,
    "looking_away": 0.3,
    "turned_to_peer": 0.3,
    "head_down": 0.15,
    "phone_use": 0.1,
    "uncertain": 0.5,
}

# face_kpts at or below this, combined with occluded=True, is visually
# (measured value removed from the handover copy)
UNCERTAIN_FACE_KPTS = 2

_TASK_ACTIVITIES = {"using_laptop", "listening", "reading", "writing_notes"}
_TASK_GAZES = {"laptop", "teacher_or_board", "own_desk", "down"}
_TASK_TARGETS = {"device", "instruction", "own_work"}


#: Rule versions. ``v1`` is the rule set every published number and every
#: built sequence was produced under and MUST NOT change. ``v2`` is the
#: repaired rule set; see RULESET_V2_RATIONALE.
RULESETS: Tuple[str, ...] = ("v1", "v2")
DEFAULT_RULESET: str = "v1"

RULESET_V2_RATIONALE = """
v2 removes ``attention_target`` from every cue rule.

Measured, on the 1,000-crop stratified sample
(grounding_data/llmstu_tools/outputs/gold_candidates.jsonl):

  * ``attention_target`` is 96.0% predictable from ``gaze_direction`` alone --
    laptop->device 100%, own_desk->own_work 100%, peer->peer 100%,
    phone->device 100%, teacher_or_board->instruction 98.8%,
    away_or_window->distracted 98.4%. It is a recoding of the gaze field, not
    an independent observation.
  * Its ONE non-redundant cell is the defect: ``gaze_direction == "down"``
    maps to ``attention_target == "distracted"`` on 94.1% of records, and
    ``distracted`` is a v1 trigger for ``looking_away``. So *looking down*
    fires *looking away*.
  * Of the 372 records where the v1 ``looking_away`` rule fires, 359 (96.5%)
    are driven by ``distracted`` and 181 (48.7%) by ``distracted`` ALONE. Of
    those 181: gaze is ``down`` on 85.1% and ``away_or_window`` on 0.0%;
    activity is ``head_down_sleeping`` on 70.7%; and 81.2% are labelled
    ``head_down`` once precedence is applied.

The same pathology is present in the human labels, so it is a property of the
annotation vocabulary rather than of the VLM pseudo-labeller. On the 754
usable human-annotated crops in ``event_gold_bundle/gold_annotations_Admin.jsonl``:
``gaze == down`` -> ``distracted`` on 231/232 = 99.6%, and 231/232 = 99.6% of
human ``head_down`` records also carry ``looking_away`` as a candidate --
reproducing the 772-of-773 co-occurrence that collapsed PRODEN [internal notes, not included]
in labels the pseudo-labeller never touched.

There is no ``attention_target`` value meaning "looking down at own work", so a
head-down student can only be called ``distracted``; the cue rule then turns
that into ``looking_away``, which sits directly below ``head_down`` in
precedence and therefore inherits exactly the frames it cannot be told apart
from. ``looking_away`` is, roughly half the time, a synonym for "head is down".

v2 therefore drops the field and keeps only perceptual evidence:

    looking_away     gaze == away_or_window OR activity == looking_away
    turned_to_peer   activity == talking_to_peer OR talking OR gaze == peer
    screen_oriented  activity in TASK_ACTIVITIES OR gaze in TASK_GAZES
    no-signal gate   gaze/engagement/activity unknown-ish (target dropped)

``target == peer`` is dropped as exactly redundant with ``gaze == peer``
(129/129 co-occurrence). ``screen_oriented`` loses the ``target in
TASK_TARGETS`` conjunct, which was only ever a restatement of the gaze test.
Dropping ``target == unknown`` from the no-signal gate makes the gate fire on
crops with no readable orientation that v1 sent to ``looking_away`` via
``distracted``; an unreadable crop becomes ``uncertain``, which is what the
annotation guideline says it is.

Effect on the stratified sample: 26/1000 hard labels change; ``looking_away``
purity (share of the class whose gaze is actually ``away_or_window``) rises
74.1% -> 90.6%; ``head_down`` records ambiguous with ``looking_away`` fall
100% -> 26.1%; the under-crediting ratio falls 2.60x -> 1.63x.

These are SAMPLE numbers, and the sample is stratified (rare activities are
oversampled), so they are not corpus prevalences. Re-measure corpus-wide with
``tools/audit_attention_target.py`` before citing any of them.
"""


def cue_conditions(rec: Dict,
                   ruleset: str = DEFAULT_RULESET) -> "List[Tuple[str, bool]]":
    """Every cue rule and whether it fires, in precedence order.

    The single source of truth for both :func:`map_record` (first match wins)
    and :func:`candidate_set` (all matches). They must not be written twice: a
    drift between the label a frame is given and the set it is credited for
    would be invisible and would silently change what the model is scored on.

    ``("uncertain", True)`` in first position is a GATE, not a candidate among
    others: a student who cannot be seen supports no cue at all, so both
    callers stop there.

    ``ruleset`` selects the rule version. The two versions share this one
    function for the same reason ``map_record`` and ``candidate_set`` do: two
    copies of a precedence list drift silently. v1 is the default everywhere,
    so no existing caller changes behaviour.
    """
    if ruleset not in RULESETS:
        raise KeyError(f"unknown ruleset {ruleset!r}; known: {RULESETS}")
    activity = rec.get("activity", "other")
    gaze = rec.get("gaze_direction", "unknown")
    target = rec.get("attention_target", "unknown")
    posture = rec.get("posture", "unknown")
    hand = rec.get("hand_state", "unknown")
    occluded = bool(rec.get("occluded", False))
    face_kpts = int(rec.get("face_kpts", 3))
    phone_visible = bool(rec.get("phone_visible", False))
    talking = bool(rec.get("talking", False))
    engagement = rec.get("engagement_level", "unknown")

    unverifiable = occluded and face_kpts <= UNCERTAIN_FACE_KPTS

    # Rules shared by both versions. phone_use and head_down never referenced
    # attention_target, so v2 leaves them untouched -- the repair is confined
    # to the three rules that read it plus the no-signal gate.
    phone = (activity == "using_phone" or phone_visible
             or gaze == "phone" or hand == "on_phone")
    head_down = (activity == "head_down_sleeping"
                 or posture in ("head_down", "slumped"))

    if ruleset == "v1":
        no_signal = (gaze == "unknown" and target == "unknown"
                     and engagement == "unknown" and activity == "other")
        peer = (activity == "talking_to_peer" or talking
                or gaze == "peer" or target == "peer")
        away = (gaze == "away_or_window" or activity == "looking_away"
                or target == "distracted")
        screen = (activity in _TASK_ACTIVITIES
                  or (gaze in _TASK_GAZES and target in _TASK_TARGETS))
    else:                                            # v2 -- see RULESET_V2_RATIONALE
        no_signal = (gaze == "unknown" and engagement == "unknown"
                     and activity == "other")
        peer = activity == "talking_to_peer" or talking or gaze == "peer"
        away = gaze == "away_or_window" or activity == "looking_away"
        screen = activity in _TASK_ACTIVITIES or gaze in _TASK_GAZES

    return [
        ("uncertain", unverifiable or no_signal),
        ("phone_use", phone),
        ("head_down", head_down),
        ("turned_to_peer", peer),
        ("looking_away", away),
        ("screen_oriented", screen),
    ]


def candidate_set(rec: Dict, ruleset: str = DEFAULT_RULESET) -> "List[int]":
    """Every cue class this record supports -- the PARTIAL label.

    ``map_record`` keeps the highest-precedence firing rule and discards the
    rest. That is not a tie-break, it is a deletion: measured over the 6,816
    LLMSTU pseudo-labels, 13.7% of records fire more than one rule, and
    `looking_away` is true by its own rule 2.74x more often than precedence
    lets it be the label (1313 against 479). The model is then penalised for
    predicting a class that was, by the annotation's own fields, correct.

    Returning the set lets a partial-label objective (PRODEN, Lv et al. ICML
    2020) credit any candidate and let the pixels decide which, instead of a
    hand-written ordering deciding in advance.
    """
    conds = cue_conditions(rec, ruleset)
    if conds[0][1]:                      # the unverifiable/no-signal gate
        return [CUE_TO_ID["uncertain"]]
    fired = [CUE_TO_ID[name] for name, hit in conds[1:] if hit]
    return fired or [CUE_TO_ID["uncertain"]]


def map_record(rec: Dict, ruleset: str = DEFAULT_RULESET) -> int:
    """Map one LLMSTU label record (parsed jsonl dict) to a cue class id.

    First firing rule wins. Kept exactly as it was: every published number and
    every built sequence depends on it. ``candidate_set`` is the partial-label
    view of the same conditions.
    """
    for name, hit in cue_conditions(rec, ruleset):
        if hit:
            return CUE_TO_ID[name]
    return CUE_TO_ID["uncertain"]


def _map_record_legacy(rec: Dict) -> int:
    """The original inlined implementation, kept only as a test oracle."""
    activity = rec.get("activity", "other")
    gaze = rec.get("gaze_direction", "unknown")
    target = rec.get("attention_target", "unknown")
    posture = rec.get("posture", "unknown")
    hand = rec.get("hand_state", "unknown")
    occluded = bool(rec.get("occluded", False))
    face_kpts = int(rec.get("face_kpts", 3))
    phone_visible = bool(rec.get("phone_visible", False))
    talking = bool(rec.get("talking", False))
    engagement = rec.get("engagement_level", "unknown")

    if occluded and face_kpts <= UNCERTAIN_FACE_KPTS:
        return CUE_TO_ID["uncertain"]
    if (
        gaze == "unknown"
        and target == "unknown"
        and engagement == "unknown"
        and activity == "other"
    ):
        return CUE_TO_ID["uncertain"]

    if activity == "using_phone" or phone_visible or gaze == "phone" or hand == "on_phone":
        return CUE_TO_ID["phone_use"]

    if activity == "head_down_sleeping" or posture in ("head_down", "slumped"):
        return CUE_TO_ID["head_down"]

    if activity == "talking_to_peer" or talking or gaze == "peer" or target == "peer":
        return CUE_TO_ID["turned_to_peer"]

    if gaze == "away_or_window" or activity == "looking_away" or target == "distracted":
        return CUE_TO_ID["looking_away"]

    if activity in _TASK_ACTIVITIES or (gaze in _TASK_GAZES and target in _TASK_TARGETS):
        return CUE_TO_ID["screen_oriented"]

    return CUE_TO_ID["uncertain"]


def parse_stem_time(stem: str) -> Optional[float]:
    """Seconds-within-video from a frame stem like ``t000033_856_f000680...``."""
    if not stem.startswith("t"):
        return None
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0][1:]) + int(parts[1]) / 1000.0
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# cue9 — a SECOND projection of Layer 1, not a regrouping of cue6
# ---------------------------------------------------------------------------
#
# (measured value removed from the handover copy)
# behaviours: writing, working on a laptop, reading off the desk, and watching
# the teacher or board. As one class it is too general to be worth reporting --
# "the student is oriented at a task" is nearly the base rate of the room.
#
# Splitting it CANNOT be done with a taxonomy. Every entry in TAXONOMIES is a
# regrouping of the six cue ids, and by the time a record is `screen_oriented`
# the fields that distinguish writing from reading are gone. cue9 is therefore a
# second projection of the SAME Layer-1 schema, with its own precedence list, and
# it needs its own label build (`build_cue_labels --label-space cue9`). Features
# are untouched: only the target changes.
#
# The four on-task rules, in precedence order, and the one change to the rest:
#
#   writing_notes   activity == writing_notes
#   using_laptop    activity == using_laptop AND target in TASK_TARGETS
#   reading         gaze in {laptop, own_desk}
#   listening       gaze == teacher_or_board
#   head_down       ALSO fires on gaze == down   <- the only change to a kept rule
#
# `head_down` gaining `gaze == down` is the repair RULESET_V2_RATIONALE argues
# for, applied where it belongs. Because `head_down` sits ABOVE `looking_away` in
# precedence, a student looking down is now `head_down` rather than
# `looking_away` -- and `gaze == down` simultaneously leaves the on-task gaze set
# (`reading` is {laptop, own_desk}, not {..., down}), so the two changes agree
# instead of competing.
#
# Measured on all 283,913 labels_tracked.jsonl records:
#
#   cue6                    cue9
#   screen_oriented 75.68%  using_laptop  31.62%   <- the 75.68% splits four ways
#                           listening     30.41%
#                           reading       11.93%
#                           writing_notes  0.75%
#   looking_away     6.72%  looking_away   6.03%   <- loses gaze==down to head_down
#   head_down        5.20%  head_down      6.87%   <- +1,934 from looking_away,
#                                                     +2,786 from screen_oriented
#   uncertain        5.03%  uncertain      5.03%
#   phone_use        4.78%  phone_use      4.78%
#   turned_to_peer   2.60%  turned_to_peer 2.59%
#
# Two consequences a reader needs:
#
# (measured value removed from the handover copy)
#   * `writing_notes` is 0.75% of records (2,119, and ~1,187 after dedup). That is
#     thin. `idle_other` was retired at 16 records because its inverse-frequency
#     weight destabilised training, and writing_notes is 130x larger than that,
#     but it is still the class to watch in any cue9 result.
#
# (measured value removed from the handover copy)
# (measured value removed from the handover copy)
# overlap (a laptop user whose gaze is on the laptop fires `using_laptop` AND
# `reading`). That is a property of the split, not a defect — but it means a
# partial-label objective over cue9 is a different proposition from one over
# cue6, and the PRODEN identifiability argument in `candidate_set` would have to
# be re-checked before trying it.
#
# The `reading`/`listening` rules are gaze-only as specified. An earlier reading
# that also fell back to `activity in {reading, listening}` was measured and
# differs on 88 records of 283,913 (0.03%), all of them `uncertain` vs
# `listening`; the fallback is kept because it leaves nothing to the fallback
# class, and it is recorded here because the choice is immaterial either way.

CUE9_CLASSES: List[str] = [
    "writing_notes",
    "using_laptop",
    "reading",
    "listening",
    "looking_away",
    "head_down",
    "turned_to_peer",
    "phone_use",
    "uncertain",
]
CUE9_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CUE9_CLASSES)}

#: The on-task classes cue9 replaces `screen_oriented` with.
CUE9_ON_TASK: Tuple[str, ...] = ("writing_notes", "using_laptop", "reading",
                                 "listening")

#: Aggregate overlay score per cue9 class. The four on-task classes inherit
#: `screen_oriented`'s 1.0 -- the split is about what is VISIBLE, not about
#: ranking one on-task behaviour above another. Not a claim about any
#: individual's mental state.
CUE9_TASK_SCORE: Dict[str, float] = {
    **{c: 1.0 for c in CUE9_ON_TASK},
    "uncertain": 0.5,
    "looking_away": 0.3,
    "turned_to_peer": 0.3,
    "head_down": 0.15,
    "phone_use": 0.1,
}

_READING_GAZES = {"laptop", "own_desk"}


def cue9_conditions(rec: Dict) -> "List[Tuple[str, bool]]":
    """Every cue9 rule and whether it fires, in precedence order.

    Same contract as :func:`cue_conditions`: the single source of truth for both
    :func:`map_record_cue9` (first match wins) and :func:`candidate_set_cue9`
    (all matches), so the label a frame is given can never drift from the set it
    is credited for. ``("uncertain", True)`` in first position is a GATE.
    """
    activity = rec.get("activity", "other")
    gaze = rec.get("gaze_direction", "unknown")
    target = rec.get("attention_target", "unknown")
    posture = rec.get("posture", "unknown")
    hand = rec.get("hand_state", "unknown")
    occluded = bool(rec.get("occluded", False))
    face_kpts = int(rec.get("face_kpts", 3))
    phone_visible = bool(rec.get("phone_visible", False))
    talking = bool(rec.get("talking", False))
    engagement = rec.get("engagement_level", "unknown")

    unverifiable = occluded and face_kpts <= UNCERTAIN_FACE_KPTS
    no_signal = (gaze == "unknown" and target == "unknown"
                 and engagement == "unknown" and activity == "other")

    return [
        ("uncertain", unverifiable or no_signal),
        ("phone_use", activity == "using_phone" or phone_visible
         or gaze == "phone" or hand == "on_phone"),
        # the one changed rule: gaze == down is a head down, not a look away
        ("head_down", activity == "head_down_sleeping"
         or posture in ("head_down", "slumped") or gaze == "down"),
        ("turned_to_peer", activity == "talking_to_peer" or talking
         or gaze == "peer" or target == "peer"),
        ("looking_away", gaze == "away_or_window" or activity == "looking_away"
         or target == "distracted"),
        # screen_oriented, split four ways. Two are keyed on `activity`, two on
        # `gaze`, exactly as specified; the `activity` disjuncts on reading and
        # listening are a fallback for a record whose gaze is unreadable, and
        # they change 88 records of 283,913 because nearly every such record
        # fires an earlier rule first.
        ("writing_notes", activity == "writing_notes"),
        ("using_laptop", activity == "using_laptop" and target in _TASK_TARGETS),
        ("reading", gaze in _READING_GAZES or activity == "reading"),
        ("listening", gaze == "teacher_or_board" or activity == "listening"),
    ]


def map_record_cue9(rec: Dict) -> int:
    """Map one LLMSTU label record to a cue9 class id. First rule wins."""
    for name, hit in cue9_conditions(rec):
        if hit:
            return CUE9_TO_ID[name]
    return CUE9_TO_ID["uncertain"]


def candidate_set_cue9(rec: Dict) -> "List[int]":
    """Every cue9 class this record supports -- the PARTIAL label."""
    conds = cue9_conditions(rec)
    if conds[0][1]:                      # the unverifiable/no-signal gate
        return [CUE9_TO_ID["uncertain"]]
    fired = [CUE9_TO_ID[name] for name, hit in conds[1:] if hit]
    return fired or [CUE9_TO_ID["uncertain"]]


#: The label spaces a taxonomy may be defined over. A taxonomy regroups the
#: classes of ONE of these; a sidecar label set built by `build_cue_labels`
#: records which one it holds, and `data.load_split` refuses a mismatch.
LABEL_SPACES: Dict[str, List[str]] = {
    "cue6": CUE_CLASSES,
    "cue9": CUE9_CLASSES,
}
DEFAULT_LABEL_SPACE: str = "cue6"

#: Per-space mappers, so a caller can build either label set by name.
LABEL_SPACE_MAPPER = {
    "cue6": map_record,
    "cue9": lambda rec, ruleset=None: map_record_cue9(rec),
}
LABEL_SPACE_CANDIDATES = {
    "cue6": candidate_set,
    "cue9": lambda rec, ruleset=None: candidate_set_cue9(rec),
}


def label_space_classes(space: str) -> List[str]:
    if space not in LABEL_SPACES:
        raise KeyError(f"unknown label space {space!r}; "
                       f"known: {', '.join(sorted(LABEL_SPACES))}")
    return list(LABEL_SPACES[space])


# ---------------------------------------------------------------------------
# Coarser taxonomies
# ---------------------------------------------------------------------------
#
# The 6 cue classes are a *projection* of the 10-field LLMSTU schema, not the
# schema itself, and two of them are projections the labels cannot support:
#
#   looking_away    fires on gaze == away_or_window OR activity == looking_away
#                   OR attention_target == distracted
#   turned_to_peer  fires on activity == talking_to_peer OR talking OR
#                   gaze == peer OR attention_target == peer
#
# Each is a disjunction of semantically different conditions, and both rest on
# the fine gaze distinctions a VLM pseudo-labeller is worst at. The measured
# consequences, on ff_det/mstcn_553_ff_s42 validation:
#
# (measured value removed from the handover copy)
#   * AUPRC lift over base rate 2.3x and 4.7x, against 11-14x elsewhere;
#   * adding the causally-correct feature -- head yaw/pitch/roll, 553 -> 556 --
# (measured value removed from the handover copy)
# (measured value removed from the handover copy)
#     target is noisy, not that the model lacks information.
#
# So `_reliable` taxonomies map those two frames to IGNORE_LABEL rather than
# forcing them into a class. They then leave the loss AND the metrics, which is
# the honest accounting: the model is not scored on them because it is not
# asked to predict them. At runtime the deployed system must abstain on them
# (measured value removed from the handover copy)
# Coverage is therefore reported alongside every _reliable number.

#: Matches attention.thesis_eval.data.IGNORE_INDEX (torch's default
#: CrossEntropyLoss ignore_index), so excluded frames vanish from the loss.
IGNORE_LABEL: int = -100

TAXONOMIES: Dict[str, Dict] = {
    "cue6": {
        "classes": CUE_CLASSES,
        "groups": {c: [c] for c in CUE_CLASSES},
        "note": "the original 6-class projection",
    },
    "onoff": {
        "classes": ["on_task", "off_task"],
        "groups": {
            "on_task": ["screen_oriented"],
            "off_task": ["looking_away", "head_down", "turned_to_peer",
                         "phone_use", "uncertain"],
        },
        "note": "binary, every frame kept; the on/off boundary IS the "
                "looking_away boundary, so this inherits its label noise",
    },
    "onoff_reliable": {
        "classes": ["on_task", "off_task"],
        "groups": {
            "on_task": ["screen_oriented"],
            "off_task": ["head_down", "phone_use", "uncertain"],
        },
        "note": "binary with abstention on the two unsupported classes",
    },
    "coarse3_reliable": {
        "classes": ["screen_oriented", "down_or_hidden", "phone_use"],
        "groups": {
            "screen_oriented": ["screen_oriented"],
            "down_or_hidden": ["head_down", "uncertain"],
            "phone_use": ["phone_use"],
        },
        "note": "keeps the actionable distinction between a head down and a "
                "phone, still abstaining on the gaze-ambiguous classes",
    },
    # Defined over the cue9 space, and a passthrough of it. It is in TAXONOMIES
    # so that --taxonomy cue9 selects it the way every other target is selected;
    # it regroups nothing, and `space` is what tells the loader that its ids
    # index CUE9_CLASSES rather than CUE_CLASSES.
    # cue9 with `reading` and `listening` folded into `using_laptop`.
    #
    # A REGROUPING of cue9, not a new projection, so it needs no label build:
    # the cue9 sidecar already carries the ids and `taxonomy_lut` merges them.
    # That is the difference between this and cue9 itself -- splitting
    # `screen_oriented` needed the annotation record back, merging three classes
    # needs only the three ids.
    #
    # The merge is defensible on the measured numbers: at 240 epochs cue9 scores
    # (measured value removed from the handover copy)
    # `reading` is keyed on `gaze in {laptop, own_desk}` while `using_laptop` is
    # keyed on `activity == using_laptop` -- the same student at a laptop can
    # satisfy either depending on which field the annotator filled. Merging them
    # removes a distinction the labels do not reliably carry.
    #
    # (measured value removed from the handover copy)
    # which makes `using_laptop` the majority class again (~74%). Expect the
    # macro-F1 to rise for the arithmetic reason that 7 classes is an easier
    # average than 9, NOT because the model improved: it is a different task,
    # and the same comparability rule applies as everywhere else here.
    "cue7": {
        "space": "cue9",
        "classes": ["writing_notes", "using_laptop", "looking_away",
                    "head_down", "turned_to_peer", "phone_use", "uncertain"],
        "groups": {
            "writing_notes": ["writing_notes"],
            "using_laptop": ["using_laptop", "reading", "listening"],
            "looking_away": ["looking_away"],
            "head_down": ["head_down"],
            "turned_to_peer": ["turned_to_peer"],
            "phone_use": ["phone_use"],
            "uncertain": ["uncertain"],
        },
        "note": "cue9 with reading and listening merged into using_laptop",
    },
    # Requested 2026-09-12: keep `using_laptop` distinct, but merge `reading`
    # and `listening` -- the two on-task cues that are NOT device-mediated --
    # into one class named `engaged`.
    #
    # Different from cue7, which folds all three into `using_laptop`. Here the
    # question is "is this student engaged with the lesson", separately from
    # "is this student working on a device", and cue9's own per-class numbers
    # (measured value removed from the handover copy)
    # (measured value removed from the handover copy)
    # whether the boundary between them was the difficulty.
    #
    # A regrouping of the cue9 SPACE, like cue7, so it needs no label rebuild:
    # the cue9 sidecar already carries the ids and `taxonomy_lut` merges them.
    "cue8": {
        "space": "cue9",
        "classes": ["writing_notes", "using_laptop", "engaged", "looking_away",
                    "head_down", "turned_to_peer", "phone_use", "uncertain"],
        "groups": {
            "writing_notes": ["writing_notes"],
            "using_laptop": ["using_laptop"],
            "engaged": ["reading", "listening"],
            "looking_away": ["looking_away"],
            "head_down": ["head_down"],
            "turned_to_peer": ["turned_to_peer"],
            "phone_use": ["phone_use"],
            "uncertain": ["uncertain"],
        },
        "note": "cue9 with reading and listening merged into `engaged`",
    },
    "cue9": {
        "space": "cue9",
        "classes": CUE9_CLASSES,
        "groups": {c: [c] for c in CUE9_CLASSES},
        "note": "screen_oriented split into writing_notes / using_laptop / "
                "reading / listening; head_down also fires on gaze == down",
    },
}


def taxonomy_space(name: str) -> str:
    """Which label space this taxonomy's groups are defined over."""
    if name not in TAXONOMIES:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    return str(TAXONOMIES[name].get("space", DEFAULT_LABEL_SPACE))


def taxonomy_classes(name: str) -> List[str]:
    if name not in TAXONOMIES:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    return list(TAXONOMIES[name]["classes"])


def taxonomy_lut(name: str) -> List[int]:
    """6-class id -> new id, or IGNORE_LABEL for a class this taxonomy drops.

    Every one of the 6 source classes must be accounted for: mapped into a
    group, or deliberately excluded. A class that is silently neither would be
    a relabelling bug that shows up only as a quietly better score.
    """
    spec = TAXONOMIES[name] if name in TAXONOMIES else None
    if spec is None:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    space = taxonomy_space(name)
    src_classes = LABEL_SPACES[space]
    src_to_id = {c: i for i, c in enumerate(src_classes)}
    lut = [IGNORE_LABEL] * len(src_classes)
    for new_id, gname in enumerate(spec["classes"]):
        for src in spec["groups"][gname]:
            if src not in src_to_id:
                raise KeyError(
                    f"taxonomy {name!r} groups {src!r}, which is not a class of "
                    f"its label space {space!r} ({', '.join(src_classes)}). A "
                    f"taxonomy may only regroup classes that exist.")
            lut[src_to_id[src]] = new_id
    return lut


def taxonomy_excluded(name: str) -> List[str]:
    """Source classes this taxonomy abstains on (mapped to IGNORE_LABEL)."""
    lut = taxonomy_lut(name)
    src = LABEL_SPACES[taxonomy_space(name)]
    return [c for i, c in enumerate(src) if lut[i] == IGNORE_LABEL]


# ---------------------------------------------------------------------------
# Projecting per-cue policy onto a regrouped taxonomy
# ---------------------------------------------------------------------------
#
# The dashboard holds two policies keyed by the SIX cue classes: which cues
# count as off-task for the classroom aggregate, and how long each must persist
# before it may page an instructor (``ALERT_AFTER_S`` in
# tools/dashboard/server.py). A model trained on `onoff_reliable` predicts
# `on_task`/`off_task`, which appear in neither, so both policies silently
# degraded to "nothing is off-task, nothing may alert" — a 2-class model would
# have shown 0% off-task on a room full of phones.
#
# These two functions project the cue-level policy onto whichever taxonomy is
# running. They deliberately use DIFFERENT rules, because the two questions are
# different:
#
#   off-task share   an aggregate. A merged class counts when it contains an
#                    off-task cue and no on-task one.
#   alert dwell      interrupts a person. A merged class may alert only when
#                    EVERY cue it merges is independently alertable, so a class
#                    that mixes an actionable cue with `uncertain` pages nobody.
#
# The consequence is deliberate and is the honest reading: `onoff_reliable`'s
# `off_task` merges `head_down` and `phone_use` (actionable) with `uncertain`
# (a student who cannot be seen), so it contributes to the off-task share and
# may raise no alerts at all. `coarse3_reliable` keeps `phone_use` as a
# singleton and can alert on it, while `down_or_hidden` cannot.

#: Cues that count as off-task for the classroom aggregate overlay.
#: `uncertain` is NEITHER — an unverifiable crop is not evidence of being
#: off-task — so it appears in no list here. cue9 splits only the ON-task side,
#: so the off-task four are identical in both spaces.
OFF_TASK_CUES: Tuple[str, ...] = ("looking_away", "head_down",
                                  "turned_to_peer", "phone_use")
ON_TASK_CUES: Tuple[str, ...] = ("screen_oriented",)

OFF_TASK_BY_SPACE: Dict[str, Tuple[str, ...]] = {
    "cue6": OFF_TASK_CUES,
    "cue9": OFF_TASK_CUES,
}
ON_TASK_BY_SPACE: Dict[str, Tuple[str, ...]] = {
    "cue6": ON_TASK_CUES,
    "cue9": CUE9_ON_TASK,
}


def taxonomy_off_task_classes(name: str) -> List[str]:
    """Classes of ``name`` that count toward the off-task share.

    A class qualifies when it merges at least one off-task cue and no on-task
    one. For ``cue6`` this returns exactly :data:`OFF_TASK_CUES`, so the
    existing behaviour is unchanged.
    """
    spec = TAXONOMIES[name] if name in TAXONOMIES else None
    if spec is None:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    space = taxonomy_space(name)
    off_cues = OFF_TASK_BY_SPACE[space]
    on_cues = ON_TASK_BY_SPACE[space]
    out = []
    for cls in spec["classes"]:
        src = spec["groups"][cls]
        if any(c in off_cues for c in src) and \
                not any(c in on_cues for c in src):
            out.append(cls)
    return out


def taxonomy_off_task_is_impure(name: str) -> Dict[str, List[str]]:
    """Off-task classes that also merge ``uncertain``, and what they merge.

    Such a class inflates the off-task share with students who merely could not
    be seen, so its percentage is NOT comparable to ``cue6``'s. The UI quotes
    this next to the number rather than leaving the two looking alike.
    """
    spec = TAXONOMIES[name]
    return {cls: list(spec["groups"][cls])
            for cls in taxonomy_off_task_classes(name)
            if "uncertain" in spec["groups"][cls]}


def taxonomy_alert_dwell(name: str,
                         cue_dwell: Dict[str, float]) -> Dict[str, float]:
    """Sustained-dwell thresholds for ``name``, projected from per-cue ones.

    ``cue_dwell`` is the cue-level policy (``server.ALERT_AFTER_S``). A merged
    class is included only when every cue it merges has a threshold, and then
    takes the LONGEST of them: merging two cues makes the evidence weaker, so
    the wait gets longer, never shorter.

    A class absent from the result may never raise an alert. That is the
    intended outcome for any class merging ``uncertain``.
    """
    spec = TAXONOMIES[name] if name in TAXONOMIES else None
    if spec is None:
        raise KeyError(f"unknown taxonomy {name!r}; "
                       f"known: {', '.join(sorted(TAXONOMIES))}")
    out: Dict[str, float] = {}
    on_cues = ON_TASK_BY_SPACE[taxonomy_space(name)]
    for cls in spec["classes"]:
        src = spec["groups"][cls]
        if not src or any(c in on_cues for c in src):
            continue
        if all(c in cue_dwell for c in src):
            out[cls] = max(cue_dwell[c] for c in src)
    return out
