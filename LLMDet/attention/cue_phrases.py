"""Natural-language phrases for the six cue classes, for querying a VLM.

Every phrase is derived from the conditions in ``taxonomy.cue_conditions`` --
the same rules that produced the labels -- rather than invented. That matters
for two reasons:

* a phrase that describes something the label does not mean would measure
  prompt-writing, not the model;
* the mapping has to be auditable, because a fused system's disagreement rate
  is only interpretable if both halves are answering the same question.

The phrases deliberately describe *visible behaviour*, matching the project's
standing rule: we claim "the student's head is down", never "the student is not
paying attention". A VLM asked to judge attention would answer a different and
less checkable question.

Known limitation, stated rather than hidden: `screen_oriented` covers both
screen and instructor orientation (taxonomy.py documents this -- in a computer
lab both are on-task), so its phrase has to name both. That makes it the
broadest query of the six, and the one most likely to absorb probability mass
from a VLM that is unsure.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .taxonomy import CUE_CLASSES

#: cue -> a phrase describing what is VISIBLE when that cue fires.
#: Each is traceable to the enum values its rule tests in cue_conditions().
CUE_PHRASES: Dict[str, str] = {
    # activity in {using_laptop, listening, reading, writing_notes}
    # or gaze in {laptop, teacher_or_board, own_desk, down} with a task target
    "screen_oriented":
        "a student facing their laptop, the teacher, the board, or their own "
        "desk, working or listening",
    # gaze == away_or_window, activity == looking_away, target == distracted
    "looking_away":
        "a student looking away from their work, turned toward a window or "
        "off into the room",
    # activity == head_down_sleeping, posture in {head_down, slumped}
    "head_down":
        "a student with their head down on the desk, slumped over or asleep",
    # activity == talking_to_peer, talking, gaze == peer, target == peer
    "turned_to_peer":
        "a student turned toward a classmate beside them, talking to them",
    # activity == using_phone, phone_visible, gaze == phone, hand == on_phone
    "phone_use":
        "a student holding a mobile phone or looking down at a phone in "
        "their hands",
    # occluded with no recoverable face signal, or no usable orientation cue
    "uncertain":
        "a student who is blocked from view, turned away, or too unclear to "
        "judge",

    # --- cue9 / cue7: the four classes screen_oriented was split into -------
    # Each is the same kind of sentence as the six above: what a person would
    # SEE, never what the student is thinking. The four are deliberately
    # contrastive -- a VLM asked to choose between them needs the difference
    # spelled out, because "working at a desk" describes all four.
    # activity == writing_notes
    "writing_notes":
        "a student writing or taking notes by hand, pen or pencil in hand, "
        "looking down at paper",
    # activity == using_laptop with a task-consistent target
    "using_laptop":
        "a student working at an open laptop or computer screen, hands at the "
        "keyboard or trackpad",
    # gaze in {laptop, own_desk}
    # cue8 merges `reading` and `listening`; the phrase has to describe the
    # union without naming a device, or it would collide with `using_laptop`.
    "engaged":
        "a student attending to the lesson without a phone or a device in "
        "hand -- reading from the desk, or watching and listening to the "
        "teacher or the board",
    "reading":
        "a student reading from a book, sheet or screen on the desk in front "
        "of them, hands not writing or typing",
    # gaze == teacher_or_board
    "listening":
        "a student facing the teacher or the board at the front of the room, "
        "attending to them rather than to their own desk",
}


def phrases_in_class_order(classes: "Optional[Sequence[str]]" = None) -> List[str]:
    """Phrases ordered to match ``classes``, so index i is class i.

    The scorer returns a vector aligned with this order and the fusion layer
    indexes it by class id. A mismatch would silently pair every cue with the
    wrong phrase, so the order is derived from the class list rather than from
    the literal above.

    ``classes`` defaults to the six cue classes. Pass ``CUE9_CLASSES`` (or any
    taxonomy's classes) to score a different label space; a class with no phrase
    raises rather than being skipped, because a short option list would shift
    every letter after it onto the wrong cue.
    """
    cl = list(CUE_CLASSES if classes is None else classes)
    missing = [c for c in cl if c not in CUE_PHRASES]
    if missing:
        raise KeyError(
            f"no cue phrase for {missing}. Every class needs one: the options "
            f"are presented as a lettered list and a gap would shift every "
            f"letter after it onto the wrong cue.")
    return [CUE_PHRASES[c] for c in cl]


def describe_mapping() -> str:
    """Human-readable dump, for the dashboard and for the thesis appendix."""
    return "\n".join(f"{c:16} -> {CUE_PHRASES[c]}" for c in CUE_CLASSES)
