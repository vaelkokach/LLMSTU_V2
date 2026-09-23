"""Every class of every taxonomy must have its own overlay colour.

This is not cosmetic. `colour_for` falls back to grey, and grey is what an
ABSTENTION draws as -- so a class with no entry renders identically to "the
model is not confident enough to say". When cue9 shipped without colours for the
four classes `screen_oriented` was split into, that was ~74% of students drawing
in the abstention colour, and the split looked like it had not happened.

The chip CSS in index.html carries the same mapping for the table, and drifting
apart would mean the box and the chip disagreed about the same student.
"""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

from attention.taxonomy import TAXONOMIES, taxonomy_classes   # noqa: E402

SR = pytest.importorskip("session_replay")
INDEX = REPO / "tools" / "dashboard" / "index.html"


@pytest.mark.parametrize("taxonomy", sorted(TAXONOMIES))
def test_every_class_has_an_overlay_colour(taxonomy):
    missing = [c for c in taxonomy_classes(taxonomy) if c not in SR.CUE_COLOUR]
    assert not missing, (
        f"{taxonomy} classes {missing} have no colour, so they draw in the "
        f"fallback grey -- which is the abstention colour. Add them to "
        f"CUE_COLOUR and to the chip CSS in index.html.")


@pytest.mark.parametrize("taxonomy", sorted(TAXONOMIES))
def test_every_class_has_a_chip_style(taxonomy):
    css = INDEX.read_text(encoding="utf-8")
    missing = [c for c in taxonomy_classes(taxonomy)
               if f".c-{c}" not in css]
    assert not missing, (
        f"{taxonomy} classes {missing} have no .c-<class> rule in index.html, "
        f"so the table chip renders unstyled while the box is coloured.")


def test_no_on_task_class_wears_an_off_task_colour():
    """The palette carries meaning: green/teal/blue is on-task, amber and red
    are off-task, grey is abstention. A class on the wrong side of that would
    be worse than having no colour at all."""
    from attention.taxonomy import (OFF_TASK_BY_SPACE, ON_TASK_BY_SPACE,
                                    taxonomy_space)
    OFF_COLOURS = {(86, 95, 255), (84, 180, 255)}
    for taxonomy in TAXONOMIES:
        space = taxonomy_space(taxonomy)
        for c in taxonomy_classes(taxonomy):
            src = TAXONOMIES[taxonomy]["groups"][c]
            # purely on-task groups only
            if not all(x in ON_TASK_BY_SPACE[space] for x in src):
                continue
            assert SR.CUE_COLOUR[c] not in OFF_COLOURS, (
                f"{taxonomy}/{c} is on-task but wears an off-task colour")


def test_abstention_draws_grey_and_nothing_else_does():
    """`uncertain` is the only class allowed the grey a fallback uses."""
    grey = SR.UNKNOWN_COLOUR
    wearers = {c for c, v in SR.CUE_COLOUR.items() if v == grey}
    assert wearers == {"uncertain"}, (
        f"{wearers - {'uncertain'}} share the abstention grey, so an "
        f"abstention and a real prediction look the same")


def test_unknown_class_still_gets_a_colour_rather_than_crashing():
    assert SR.colour_for("a_class_that_does_not_exist") == SR.UNKNOWN_COLOUR
