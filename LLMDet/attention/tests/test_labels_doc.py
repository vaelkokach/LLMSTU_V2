"""`docs/LABELS.md` must agree with the code it transcribes.

A label inventory is only worth having if it is true, and a document that
merely *looks* authoritative is worse than none: the whole point of writing the
vocabulary down in one place is that a reader stops opening `vocab.py`. So this
does not check that the document mentions things — it checks set equality in
both directions, which is what catches the failure that actually happens:
someone adds or removes a value in the code and the prose quietly keeps
describing the old world.

`TAXONOMY.md` is the cautionary example. It still presents `idle_other` as a
live seventh class in its main table, eleven months after the merge into
`uncertain`, because nothing was ever checking.

Cheap and offline: no torch, no mmcv, no data.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from attention.taxonomy import (CUE_CLASSES, CUE_TASK_SCORE, IGNORE_LABEL,
                                RULESETS, TAXONOMIES, taxonomy_classes,
                                taxonomy_excluded)

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / "docs" / "LABELS.md"
VOCAB = REPO / "tools" / "gold_annotator" / "vocab.py"


def _load_vocab():
    """Import vocab.py by path -- it is not on the `attention` import path."""
    spec = importlib.util.spec_from_file_location("_gold_vocab", VOCAB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def doc() -> str:
    if not DOC.exists():
        pytest.fail(f"{DOC} is missing; it is the label inventory this suite guards")
    return DOC.read_text(encoding="utf-8")


def _row(doc: str, first_cell: str) -> str:
    """The markdown table row whose first cell is `` `first_cell` ``."""
    for line in doc.splitlines():
        if line.startswith(f"| `{first_cell}` |"):
            return line
    raise AssertionError(f"no table row for {first_cell!r} in {DOC.name}")


def _ticked(s: str) -> set:
    return set(re.findall(r"`([^`]+)`", s))


# ---------------------------------------------------------------------------
# Layer 1 -- the annotation vocabulary
# ---------------------------------------------------------------------------

def test_every_categorical_field_has_a_row_with_exactly_its_values(doc):
    """Set equality, not membership: a value dropped from vocab.py must not
    survive in the document, and a new one must not be missing from it."""
    vocab = _load_vocab()
    for field, values in vocab.CATEGORICAL_FIELDS.items():
        row = _row(doc, field)
        # The row's first cell is the field name itself; the rest are values.
        got = _ticked(row) - {field}
        assert got == set(values), (
            f"{field}: document lists {sorted(got)}, vocab.py defines "
            f"{sorted(values)}")


def test_declared_value_counts_match(doc):
    """The `n` column is a second, independent statement of the same fact, so
    it is worth checking -- a hand-edited row often updates one and not both."""
    vocab = _load_vocab()
    for field, values in vocab.CATEGORICAL_FIELDS.items():
        cells = [c.strip() for c in _row(doc, field).strip("|").split("|")]
        assert cells[-1] == str(len(values)), (
            f"{field}: document says n={cells[-1]}, vocab.py has "
            f"{len(values)} values")


def test_boolean_fields_are_all_named(doc):
    vocab = _load_vocab()
    for f in vocab.BOOLEAN_FIELDS:
        assert f"`{f}`" in doc, f"boolean field {f} is not in {DOC.name}"


def test_no_field_is_missing_from_the_document(doc):
    vocab = _load_vocab()
    for f in vocab.ALL_LABEL_FIELDS:
        assert f"`{f}`" in doc, f"field {f} is not in {DOC.name}"


# ---------------------------------------------------------------------------
# Layer 2 -- the cue classes
# ---------------------------------------------------------------------------

def test_every_cue_class_has_a_rule_row_at_its_own_id(doc):
    """The id column is load-bearing: these ids index model outputs, so a
    document that renumbers them would mislabel every prediction a reader
    traced by hand."""
    for i, cue in enumerate(CUE_CLASSES):
        want = f"| {i} | `{cue}` |"
        assert want in doc, f"expected a rule row {want!r} in {DOC.name}"


def test_no_retired_cue_is_presented_as_live(doc):
    """`idle_other` merged into `uncertain` on 2026-07-29. It may be named in
    the 'Retired class' section, but never as one of the six."""
    for i in range(len(CUE_CLASSES) + 2):
        assert f"| {i} | `idle_other` |" not in doc
    assert "idle_other" not in CUE_CLASSES


def test_cue_task_scores_match(doc):
    for cue, score in CUE_TASK_SCORE.items():
        want = f"| `{cue}` | {score:.2f} |"
        assert want in doc, (
            f"CUE_TASK_SCORE[{cue}] is {score}; expected row {want!r}")


def test_both_rulesets_are_named(doc):
    for r in RULESETS:
        assert f"**{r}**" in doc, f"ruleset {r} is not described in {DOC.name}"


# ---------------------------------------------------------------------------
# Layer 3 -- the regrouped taxonomies
# ---------------------------------------------------------------------------

def test_every_taxonomy_has_a_row_listing_its_classes_and_exclusions(doc):
    for name in TAXONOMIES:
        row = _row(doc, name)
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) >= 3, f"{name}: row is missing columns"
        listed = _ticked(cells[1])
        assert listed == set(taxonomy_classes(name)), (
            f"{name}: document lists classes {sorted(listed)}, "
            f"TAXONOMIES has {sorted(taxonomy_classes(name))}")
        excluded = _ticked(cells[2])
        assert excluded == set(taxonomy_excluded(name)), (
            f"{name}: document says it abstains on {sorted(excluded)}, "
            f"taxonomy_excluded gives {sorted(taxonomy_excluded(name))}")


def test_the_ignore_sentinel_is_stated(doc):
    assert f"IGNORE_LABEL = {IGNORE_LABEL}" in doc


def test_abstention_requires_coverage_to_be_quoted(doc):
    """Every _reliable number in this repo has to carry its coverage. If the
    document ever drops that instruction, the next person to quote [value removed]
    against [value removed] has no warning."""
    assert "coverage" in doc.lower()
