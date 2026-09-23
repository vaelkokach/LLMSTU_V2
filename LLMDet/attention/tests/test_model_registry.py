"""The dashboard registry must not average or rank across different targets.

Two runs are seeds of one variant only if they were asked the same question.
Before this was enforced, ``work_dirs/thesis/`` broke it three ways at once:

* ``coarse/`` holds ``cue6``, ``onoff_reliable`` and ``coarse3_reliable`` at one
  architecture and feature config. All seven runs grouped as one "variant"
  whose seed spread mixed 6-, 3- and 2-class macro-F1, and whose best "seed"
  was a 2-class model at [value removed]. Ranking by that made it the dashboard DEFAULT
  over the model the thesis deploys.
* ``cue_v2/`` holds two cue RULESETS x two objectives x 3 seeds — twelve runs,
  four targets, one "variant".
* ``wave2/`` mixes single-label and PRODEN runs, which is where its
  implausible seed sd of [value removed] came from.

These tests run against synthetic run records, so they hold regardless of what
is in ``work_dirs`` today, plus one test against the real tree when it is
present.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "dashboard"))

MR = pytest.importorskip("model_registry")


def _run(root, sweep, exp, *, model="mstcn", fc="556_hp", taxonomy="cue6",
         seed=42, partial=False, cue_labels="", macro_f1=0.5, coverage=1.0,
         seq_root="../grounding_data/llmstu_sequences_full_det"):
    """Write a minimal run_record + metrics + checkpoint stub."""
    d = root / sweep / exp
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "best.pth").write_bytes(b"stub")
    (d / "run_record.json").write_text(json.dumps({"spec": {
        "experiment_id": exp, "model": model, "feature_config": fc,
        "taxonomy": taxonomy, "seed": seed, "partial_labels": partial,
        "cue_labels": cue_labels, "sequence_root": seq_root}}))
    (d / "eval_val").mkdir()
    (d / "eval_val" / "metrics.json").write_text(json.dumps({
        "macro_f1": macro_f1, "accuracy": macro_f1, "coverage": coverage,
        "n_frames": 1000, "n_videos": 27}))
    return d


def test_taxonomies_do_not_group_together(tmp_path):
    """The defect, as a test: three taxonomies at one arch/config."""
    _run(tmp_path, "coarse", "cue6_s42", taxonomy="cue6", macro_f1=0.48)
    _run(tmp_path, "coarse", "onoff_s43", taxonomy="onoff_reliable",
         seed=43, macro_f1=0.786, coverage=0.91)
    _run(tmp_path, "coarse", "c3_s44", taxonomy="coarse3_reliable",
         seed=44, macro_f1=0.767, coverage=0.91)

    entries = MR.scan(tmp_path)
    assert len(entries) == 3, "three taxonomies must be three variants"
    for e in entries:
        assert e.n_seeds == 1, f"{e.variant_id} absorbed another taxonomy's run"
        assert e.val_seed_mean["macro_f1_sd"] == 0.0


def test_a_coarse_model_never_becomes_the_default(tmp_path):
    """It scores highest and must still lose: [value removed] over 2 classes at 91%
    coverage is not comparable to 0.48 over 6 at 100% (docs/LABELS.md)."""
    _run(tmp_path, "coarse", "cue6_s42", taxonomy="cue6", macro_f1=0.48)
    _run(tmp_path, "coarse", "onoff_s43", taxonomy="onoff_reliable",
         seed=43, macro_f1=0.786, coverage=0.91)

    default = MR.default_entry(MR.scan(tmp_path))
    assert default.taxonomy == "cue6"
    assert default.is_canonical
    assert default.val["macro_f1"] == 0.48


def test_rulesets_and_objectives_do_not_group_together(tmp_path):
    """cue_v2/: v1/v2 x base/proden must be four variants, not one."""
    for rs, cl in (("v1", ""), ("v2", "../grounding_data/cue_labels_v2_full_det.npz")):
        for obj in (False, True):
            for seed in (42, 43, 44):
                _run(tmp_path, "cue_v2", f"{rs}_{'proden' if obj else 'base'}_s{seed}",
                     seed=seed, partial=obj, cue_labels=cl, macro_f1=0.49)

    entries = MR.scan(tmp_path)
    assert len(entries) == 4, [e.variant_id for e in entries]
    assert {e.n_seeds for e in entries} == {3}
    assert {e.comparable_group for e in entries} == {
        "cue6/v1/single", "cue6/v1/proden", "cue6/v2/single", "cue6/v2/proden"}


def test_the_canonical_variant_id_is_unchanged(tmp_path):
    """A cue6/v1/single-label run keeps the id it has always had.

    ``attention_runtime.yaml``, the Space's DASHBOARD_MODEL variable, the fitted
    calibration filenames and every id quoted in [internal notes, not included] resolve by this
    string. Only the runs that were previously COLLIDING may get a new one.
    """
    _run(tmp_path, "arch", "mstcn_556_hp_s43", seed=43, macro_f1=0.5067)
    e, = MR.scan(tmp_path)
    assert e.variant_id == "arch/mstcn_556_hp"
    assert e.comparable_group == "cue6/v1/single"


@pytest.mark.parametrize("taxonomy,ruleset_labels,obj,want", [
    ("onoff_reliable", "", False, "coarse/mstcn_556_hp:onoff_reliable"),
    ("cue6", "../grounding_data/cue_labels_v2_full_det.npz", False,
     "coarse/mstcn_556_hp:v2"),
    ("cue6", "", True, "coarse/mstcn_556_hp:proden"),
    ("cue6", "../grounding_data/cue_labels_v2_full_det.npz", True,
     "coarse/mstcn_556_hp:v2+proden"),
])
def test_non_canonical_ids_name_their_target(tmp_path, taxonomy, ruleset_labels,
                                             obj, want):
    _run(tmp_path, "coarse", "x_s42", taxonomy=taxonomy,
         cue_labels=ruleset_labels, partial=obj)
    e, = MR.scan(tmp_path)
    assert e.variant_id == want


def test_head_stream_is_live_but_not_replayable(tmp_path):
    """1074_hp_head reads column 1073 and is perfectly live-servable: the live
    extractor grows the block from head_stream=True. A session cache stores
    base + head pose only, so it cannot replay. The old LIVE_MAX_COL=556 rule
    blocked it for being wide and hid the best validation model."""
    _run(tmp_path, "wave2", "mstcn_head_s44", fc="1074_hp_head", seed=44,
         macro_f1=0.5307)
    e, = MR.scan(tmp_path)
    assert e.live_capable and e.needs_head_stream
    assert not e.replay_capable
    assert not e.is_default, "the default must work in replay, which boots first"


@pytest.mark.parametrize("fc", ["563_expr", "563_dyn", "570_full"])
def test_expression_and_dynamics_are_undeployable(tmp_path, fc):
    """These two blocks, not a column count, are what a streaming path cannot
    produce: expression needs a per-crop FER model, dynamics is a whole-track
    statistic."""
    _run(tmp_path, "ladder", "x_s42", fc=fc)
    e, = MR.scan(tmp_path)
    assert not e.live_capable and not e.replay_capable
    assert e.blocked_reason


def test_coverage_travels_with_the_number(tmp_path):
    _run(tmp_path, "coarse", "onoff_s43", taxonomy="onoff_reliable", seed=43,
         macro_f1=0.786, coverage=0.9102)
    e, = MR.scan(tmp_path)
    assert e.coverage == pytest.approx(0.9102)
    assert e.n_classes == 2
    assert set(e.abstains_on) == {"looking_away", "turned_to_peer"}


def test_best_in_group_never_crosses_targets(tmp_path):
    _run(tmp_path, "coarse", "cue6_s42", taxonomy="cue6", macro_f1=0.48)
    _run(tmp_path, "coarse", "onoff_s43", taxonomy="onoff_reliable", seed=43,
         macro_f1=0.786, coverage=0.91)
    entries = MR.scan(tmp_path)
    assert MR.best_in_group(entries, "cue6").val["macro_f1"] == 0.48
    assert MR.best_in_group(entries, "onoff_reliable").val["macro_f1"] == 0.786
    assert MR.best_in_group(entries, "no_such_taxonomy") is None


def test_the_epoch_budget_is_not_part_of_the_target(tmp_path):
    """A longer run is the SAME question asked of a better-trained model.

    The 240-epoch family re-run [internal notes, not included] puts a second cue6/556_hp variant
    on disk. It must land in the canonical comparability group and rank against
    the 90-epoch one on validation — the budget changes how well the model
    answers, not what it was asked. Putting it in the key would split one
    question into two and let a worse model keep the default by never being
    compared to the better one.
    """
    _run(tmp_path, "coarse", "cue6_s42", macro_f1=0.4838)
    _run(tmp_path, "epochs240", "mstcn_556_cue6_s42", macro_f1=0.5100)

    entries = MR.scan(tmp_path)
    assert {e.variant_id for e in entries} == {
        "coarse/mstcn_556_hp", "epochs240/mstcn_556_hp"}
    assert {e.comparable_group for e in entries} == {"cue6/v1/single"}
    # the better-trained one wins the default, because they ARE comparable
    assert MR.default_entry(entries).variant_id == "epochs240/mstcn_556_hp"


def test_a_single_seed_probe_never_becomes_the_default(tmp_path):
    """cue9_probe is one seed at a larger budget, run to measure what the budget
    was costing. Best-of-one is a measurement, not a selection, so it is shown
    and never chosen."""
    _run(tmp_path, "coarse", "cue6_s42", macro_f1=0.4838)
    _run(tmp_path, "cue9_probe", "probe_s42", macro_f1=0.9000)

    entries = MR.scan(tmp_path)
    assert MR.default_entry(entries).sweep == "coarse"
    probe, = [e for e in entries if e.sweep == "cue9_probe"]
    assert not probe.is_default
    assert probe in entries, "a probe is still offered, just never defaulted to"


@pytest.mark.skipif(not (REPO / "LLMDet" / "work_dirs" / "thesis").is_dir(),
                    reason="needs the real work_dirs tree")
def test_the_real_tree_has_one_target_per_variant():
    """_make_entry raises on a mixed variant; scanning the real tree is the
    end-to-end check that no sweep on disk still collides."""
    entries = MR.scan()
    assert entries
    default = MR.default_entry(entries)
    assert default.is_canonical and default.live_capable and default.replay_capable
    for e in entries:
        assert e.n_classes == 0 or e.n_classes == len(e.class_names)
