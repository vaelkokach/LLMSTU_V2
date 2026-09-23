"""Which trained models the dashboard is allowed to offer, and which is default.

The sweeps under ``work_dirs/thesis/`` hold well over a hundred checkpoints.
The dashboard must not present that as a flat list of equivalent choices — most
of them are seeds of the same variant, several are trained against a *different
target*, and two of the feature configs cannot run live at all.

This module reduces the sweeps to one entry per **variant**, choosing the seed
with the best macro-F1 on the **validation** split. Selection never reads the
test split: the test numbers are carried along and displayed, but they never
decide anything. That is the same discipline as ``[internal notes, not included]`` — the
test split was spent once, and a dropdown that ranked models by it would be
spending it again, once per page load.

What makes two runs the same variant
------------------------------------
A variant used to be ``sweep x architecture x feature config``. That is not
enough, and the consequences were live:

* ``coarse/`` holds three taxonomies at one architecture and feature config
  (``cue6``, ``onoff_reliable``, ``coarse3_reliable``). All seven runs collapsed
  into one "variant" whose seed spread mixed 6-class, 3-class and 2-class
  macro-F1 ([value removed] +- [value removed] — a spread that is an artefact of averaging three
  different tasks), and whose best "seed" was a 2-class model at [value removed]. Sorting
  by that number made it the **default model of the dashboard**, over the 6-class
  model the thesis actually deploys. ``docs/LABELS.md`` states the rule it broke:
  *macro-F1 over 2 or 3 classes is not comparable to macro-F1 over 6.*
* ``cue_v2/`` holds ``v1_base``, ``v1_proden``, ``v2_base``, ``v2_proden`` x 3
  seeds — two cue RULESETS and two objectives — all at ``mstcn``/``556_hp``. Twelve
  runs, one "variant", four targets.
* ``wave2/`` mixes single-label and PRODEN runs the same way, which is where its
  implausible seed sd of [value removed] came from.

So the variant key now carries everything that changes **what the model was
asked to predict**: taxonomy, cue ruleset, and whether the objective was
partial-label. Runs that differ in any of those are different variants, are
never averaged together, and are never ranked against each other.

Comparability
-------------
``comparable_group`` names the (taxonomy, ruleset, objective) triple. Ranking and
defaulting happen *within* the canonical group only — ``cue6`` under ruleset v1
with a single-label objective, which is the target every published number in
``[internal notes, not included]`` is measured against. Other groups are offered, labelled with
their class count and coverage, and sorted among themselves; they can never
become the default by scoring high on an easier task.

Deployability
-------------
Judged by which feature BLOCKS a config reads, not by a column count. The two
blocks a streaming path cannot produce are ``express`` (needs a second per-crop
FER model) and ``dynamic`` (a whole-track statistic, not a per-frame quantity).
Everything else it can, including the 518-dim ``head`` block — the live
extractor grows it from ``head_stream=True`` (``features.py:116``), so the
head-stream model is live-capable even though it is 1074 columns wide. The old
``LIVE_MAX_COL = 556`` rule blocked it purely for being wide, which hid the best
validation model in the registry behind a column count.

Replay is a second, narrower question: a cached session holds ``base`` and both
head-pose blocks and nothing else, so a head-stream model can analyse live video
but cannot re-decide an existing session cache. The two capabilities are
reported separately rather than collapsed into one "deployable" flag, because
they fail in different places.

Three facts about each variant come from the checkpoint's own ``run_record.json``
rather than from a table maintained by hand here:

``spec.feature_config``
    which columns of the live vector the model consumes.

``spec.sequence_root``
    which head-pose backend produced the 4-column block the model was trained
    on: ``_det`` -> BlazeFace detector, everything else -> FaceLandmarker mesh.
    The session cache stores both blocks, so each model is fed the one it was
    trained with instead of whichever the last run happened to use.

``eval_val/metrics.json``
    the evaluator's own numbers (``thesis_eval/1.0.0``), so what the dropdown
    shows is what the thesis tables show, from the same file — including
    ``coverage``, which every abstaining taxonomy's number must be quoted with.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
THESIS = REPO / "LLMDet" / "work_dirs" / "thesis"

#: Feature blocks a streaming extractor can produce, and why the others cannot.
#: Keyed by the block names in ``thesis_eval.data.LAYOUTS``.
#: `objects` is live-capable but needs a SECOND detector in the path (see
#: pipeline_bridge), which is why it is not in CACHED_BLOCKS: no existing
#: session cache holds those six columns.
LIVE_BLOCKS = {"base", "headpose", "hp_angles", "hp_facefound", "head",
               "objects"}
BLOCKED_BLOCKS = {
    "express": "the 7 expression dims need a second per-crop FER model",
    "dynamic": "the 7 dynamic dims are whole-track statistics (fidget variance, "
               "a personalised gaze baseline), not per-frame quantities",
}
#: Blocks a precomputed session cache stores. It holds `base` plus both
#: head-pose variants; it does not store a head-crop CLIP pass.
CACHED_BLOCKS = {"base", "headpose", "hp_angles", "hp_facefound"}

#: Fallback block lists, used only if ``thesis_eval.data`` cannot be imported
#: (no numpy/torch). Keeps the registry buildable for the pure-stdlib replay
#: mode, at the cost of not validating a config name it does not know.
_FALLBACK_BLOCKS = {
    "552_base": ["base"],
    "553_facefound": ["base", "hp_facefound"],
    "555_angles": ["base", "hp_angles"],
    "556_hp": ["base", "headpose"],
    "563_expr": ["base", "headpose", "express"],
    "563_dyn": ["base", "headpose", "dynamic"],
    "570_full": ["base", "headpose", "express", "dynamic"],
    "1070_head": ["base", "head"],
    "1074_hp_head": ["base", "headpose", "head"],
    "1080_hp_head_obj": ["base", "headpose", "head", "objects"],
}

#: Human labels. Keys are the trainer's ``feature_config`` strings.
FEATURE_LABEL = {
    "552_base": "CLIP + geometry + colour + posture",
    "553_facefound": "+ face-found flag",
    "555_angles": "+ head-pose angles",
    "556_hp": "+ head pose (angles + flag)",
    "563_expr": "+ facial expression",
    "563_dyn": "+ motion dynamics",
    "570_full": "+ expression + dynamics",
    "1070_head": "+ head-crop CLIP stream",
    "1074_hp_head": "+ head pose + head-crop CLIP stream",
    "1080_hp_head_obj": "+ head pose + head-crop CLIP stream + phone/laptop "
                        "detections",
}

ARCH_LABEL = {"transformer": "Transformer", "mstcn": "MS-TCN", "asrf": "ASRF"}

#: What each sweep varied, so the dropdown explains why two entries share an
#: architecture and a feature config.
SWEEP_LABEL = {
    "ladder": "feature ladder",
    "arch": "architecture sweep",
    "headpose": "head-pose ablation",
    "posefix": "body-pose train/deploy fix (rejected, ref. [internal notes, not included]",
    "ff_bp": "face-found, landmarker on the person box",
    "ff_det": "face-found, BlazeFace detector",
    "coarse": "coarser taxonomies",
    "cue_v2": "cue ruleset v1 vs v2 (v2 failed, ref. docs/CUE_RULES_V2.md)",
    "wave2": "head stream + partial labels",
    "wave2b": "partial labels, reduction fixed",
    "cue9": "screen_oriented split into four on-task cues",
    "cue9_probe": "cue9 convergence probe (single seed, not a reported result)",
    "epochs240": "the full_det family re-run at 240 epochs [internal notes, not included]",
    "cue7": "reading + listening merged back into using_laptop",
    "objfeat": "phone/laptop object columns, matched against the same tree "
               "without them [internal notes, not included]",
}

#: The target every published number in [internal notes, not included] is measured against. Only
#: models trained on it may become the dashboard default.
CANONICAL_GROUP = ("cue6", "v1", "single")

#: Sweeps that exist to answer a question, not to produce a servable model. They
#: are scanned and shown, but never chosen as the default: a single-seed probe
#: is a measurement, and best-of-one is not a selection.
PROBE_SWEEPS = {"cue9_probe"}


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute when it cannot.

    ``scan()`` takes a ``thesis_root`` argument, so a caller may legitimately
    point it at a tree outside the repository — a test fixture, or a Space that
    mounted the artifacts elsewhere. ``relative_to`` raises ValueError for those,
    which turned a supported argument into a crash.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _head_pose_backend(sequence_root: str) -> str:
    """The backend whose 4-column block this model was trained on.

    ``llmstu_sequences_full_det`` was built with the BlazeFace detector
    [internal notes, not included]; ``llmstu_sequences_full`` and ``..._bp`` with the
    FaceLandmarker mesh. Feeding a model the other block would be a silent
    train/deploy mismatch of exactly the kind 11.15 was written about.
    """
    return "mediapipe_detector" if sequence_root.rstrip("/").endswith("_det") \
        else "mediapipe"


def _label_space(spec: Dict) -> str:
    """Which label space the taxonomy is defined over — "cue6" or "cue9".

    Read from the taxonomy rather than from the sidecar, because that is what
    ``data.load_split`` validates against; the two are required to agree and it
    refuses the run if they do not.
    """
    taxonomy = str(spec.get("taxonomy", "cue6") or "cue6")
    try:
        import sys
        sys.path.insert(0, str(REPO / "LLMDet"))
        from attention.taxonomy import taxonomy_space
        return taxonomy_space(taxonomy)
    except Exception:
        return "cue9" if taxonomy == "cue9" else "cue6"


def _ruleset(spec: Dict) -> str:
    """Which cue ruleset the training labels came from.

    An empty ``cue_labels`` means the labels stored in the sequence build, which
    are v1 by construction (``build_cue_labels`` proves this on every run). A
    sidecar names its ruleset in the filename, and ``CueLabels`` carries it
    inside the npz — the filename is used here so the registry stays importable
    without numpy.

    ``v1``/``v2`` are versions of the SIX-class rules. The cue9 space has a
    single precedence list and takes no ruleset at all (``build_cue_labels``
    refuses ``--ruleset v2`` with ``--label-space cue9``), so asking which
    version it used is a category error — it reports v1 rather than "unknown",
    which would otherwise leak into its variant id and its comparability group.
    """
    if _label_space(spec) != "cue6":
        return "v1"
    cl = str(spec.get("cue_labels", "") or "")
    if not cl:
        return "v1"
    stem = Path(cl).stem
    for rs in ("v2", "v1"):
        if f"_{rs}_" in stem or stem.endswith(f"_{rs}") or f"labels_{rs}" in stem:
            return rs
    return "unknown"


def _target(spec: Dict) -> Tuple[str, str, str]:
    """(taxonomy, ruleset, objective) — everything that changes the target.

    Two runs that differ here were asked different questions, so they are not
    seeds of one variant and their macro-F1 values are not comparable.
    """
    taxonomy = str(spec.get("taxonomy", "cue6") or "cue6")
    objective = "proden" if bool(spec.get("partial_labels", False)) else "single"
    return taxonomy, _ruleset(spec), objective


def _target_suffix(target: Tuple[str, str, str]) -> str:
    """Variant-id suffix for a non-canonical target; '' for the canonical one.

    Canonical ids are left byte-identical to what they have always been, so
    ``attention_runtime.yaml``, the Space's ``DASHBOARD_MODEL`` variable, the
    fitted calibration filenames under ``runtime/dashboard/`` and every id
    quoted in [internal notes, not included] keep resolving. Only the runs that were previously
    *colliding* get a new, longer id.
    """
    taxonomy, ruleset, objective = target
    parts = []
    if taxonomy != "cue6":
        parts.append(taxonomy)
    if ruleset != "v1":
        parts.append(ruleset)
    if objective != "single":
        parts.append(objective)
    return (":" + "+".join(parts)) if parts else ""


def _blocks(feature_config: str) -> List[str]:
    """Which feature blocks this config reads, in column order."""
    try:
        import sys
        sys.path.insert(0, str(REPO / "LLMDet"))
        from attention.thesis_eval import data as D
        return list(D.FEATURE_CONFIGS[feature_config])
    except Exception:
        return list(_FALLBACK_BLOCKS.get(feature_config, ["base"]))


def _capabilities(feature_config: str) -> Tuple[bool, bool, str]:
    """(live_capable, replay_capable, reason the blocked one is blocked)."""
    blocks = _blocks(feature_config)
    bad = [b for b in blocks if b in BLOCKED_BLOCKS]
    live = not bad
    replay = all(b in CACHED_BLOCKS for b in blocks)
    if bad:
        why = "; ".join(BLOCKED_BLOCKS[b] for b in bad)
        return False, False, f"cannot run live or replay: {why}"
    if not replay:
        extra = sorted(set(blocks) - CACHED_BLOCKS)
        return True, False, (
            f"live only: a session cache stores base + head pose, not "
            f"{', '.join(extra)}. Analyse a video or a camera with this model; "
            f"re-deciding an existing session cache would need the cache "
            f"rebuilt with that block.")
    return True, True, ""


@dataclass
class ModelEntry:
    variant_id: str
    label: str
    sweep: str
    sweep_label: str
    model: str
    feature_config: str
    input_dim: int
    experiment_id: str
    seed: int
    n_seeds: int
    checkpoint: str
    head_pose_backend: str
    sequence_root: str
    #: Deployability, split by where it fails. ``deployable`` is kept as an
    #: alias for live capability so existing callers keep working.
    deployable: bool
    live_capable: bool
    replay_capable: bool
    blocked_reason: str
    #: What the model was asked to predict. Two entries whose
    #: ``comparable_group`` differs must never be ranked against each other.
    taxonomy: str
    ruleset: str
    objective: str
    comparable_group: str
    is_canonical: bool
    class_names: List[str] = field(default_factory=list)
    n_classes: int = 0
    abstains_on: List[str] = field(default_factory=list)
    #: Share of validation frames the taxonomy did NOT abstain on. Every number
    #: from an abstaining taxonomy must be quoted with it (docs/LABELS.md).
    coverage: float = 1.0
    #: True when the head stream must be switched on in the live extractor.
    needs_head_stream: bool = False
    #: Set on a VIRTUAL entry: a real checkpoint plus a VLM second opinion.
    #: There is no separate checkpoint for it — `checkpoint` still points at the
    #: base model and `vlm_base` names the variant it wraps.
    vlm: bool = False
    vlm_base: str = ""
    vlm_model_id: str = ""
    vlm_policy: str = ""
    #: On the curated shortlist the dashboard shows by default — the best model
    #: for each taxonomy. Everything else stays in the registry and is one
    #: checkbox away; nothing is deleted. See :func:`_mark_recommended`.
    recommended: bool = False
    recommended_why: str = ""
    val: Dict[str, float] = field(default_factory=dict)
    test: Dict[str, float] = field(default_factory=dict)
    #: mean +/- sd of macro-F1 over the variant's seeds. The thesis tables
    #: report this; the dashboard must run ONE checkpoint, so `val`/`test` above
    #: are that single seed's numbers and read higher than the mean by
    #: construction — best-of-3 is a selection, not a measurement. Both are
    #: shown so the deployed figure is never mistaken for the reported one.
    val_seed_mean: Dict[str, float] = field(default_factory=dict)
    test_seed_mean: Dict[str, float] = field(default_factory=dict)
    val_predictions: str = ""
    calibration: str = ""
    is_default: bool = False

    def to_json(self) -> Dict:
        return asdict(self)


def _metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    m = json.loads(path.read_text())
    return {k: m[k] for k in
            ("macro_f1", "accuracy", "balanced_accuracy", "macro_auprc", "ece",
             "coverage", "n_frames", "n_videos")
            if k in m}


def _taxonomy_meta(name: str) -> Tuple[List[str], List[str]]:
    """(class names, source cues this taxonomy abstains on)."""
    try:
        import sys
        sys.path.insert(0, str(REPO / "LLMDet"))
        from attention.taxonomy import taxonomy_classes, taxonomy_excluded
        return taxonomy_classes(name), taxonomy_excluded(name)
    except Exception:
        return [], []


def _collect(thesis_root: Path) -> Dict[str, List[Dict]]:
    """Every evaluated run that has a checkpoint, grouped by variant id."""
    by_variant: Dict[str, List[Dict]] = {}
    for record in sorted(thesis_root.glob("*/*/run_record.json")):
        exp_dir = record.parent
        sweep = exp_dir.parent.name
        rec = json.loads(record.read_text())
        spec = rec["spec"]
        ckpt = exp_dir / "checkpoints" / "best.pth"
        val = _metrics(exp_dir / "eval_val" / "metrics.json")
        if not ckpt.exists() or "macro_f1" not in val:
            # No checkpoint, or never evaluated -> nothing honest to display.
            continue
        target = _target(spec)
        vid = (f"{sweep}/{spec['model']}_{spec['feature_config']}"
               f"{_target_suffix(target)}")
        by_variant.setdefault(vid, []).append({
            "sweep": sweep, "spec": spec, "dir": exp_dir, "ckpt": ckpt,
            "target": target,
            "val": val, "test": _metrics(exp_dir / "eval_test" / "metrics.json"),
        })
    return by_variant


def _seed_mean(seeds: List[Dict], split: str) -> Dict[str, float]:
    vals = [r[split]["macro_f1"] for r in seeds if r[split]]
    if not vals:
        return {}
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 \
        if len(vals) > 1 else 0.0
    return {"macro_f1": m, "macro_f1_sd": sd, "n": len(vals)}


def _make_entry(variant_id: str, seeds: List[Dict], chosen: Dict) -> ModelEntry:
    """One registry row: `chosen` supplies the checkpoint, `seeds` the spread."""
    spec = chosen["spec"]
    fc = spec["feature_config"]
    live, replay, reason = _capabilities(fc)
    seq_root = spec.get("sequence_root", "")
    taxonomy, ruleset, objective = chosen["target"]

    # Averaging across targets is the defect this key was widened to prevent;
    # assert it rather than trust it, so adding a spec field that changes the
    # target cannot silently regroup runs again.
    others = {r["target"] for r in seeds}
    if others != {chosen["target"]}:
        raise SystemExit(
            f"{variant_id} groups more than one target: {sorted(others)}. "
            f"Averaging their macro-F1 would mix different tasks — widen "
            f"_target()/_target_suffix() to separate them.")

    classes, abstains = _taxonomy_meta(taxonomy)
    label = f"{ARCH_LABEL.get(spec['model'], spec['model'])} · {fc}"
    if taxonomy != "cue6":
        label += f" · {taxonomy}"
    if ruleset != "v1":
        label += f" · ruleset {ruleset}"
    if objective != "single":
        label += " · PRODEN"

    return ModelEntry(
        variant_id=variant_id,
        label=label,
        sweep=chosen["sweep"],
        sweep_label=SWEEP_LABEL.get(chosen["sweep"], chosen["sweep"]),
        model=spec["model"],
        feature_config=fc,
        input_dim=int(fc.split("_")[0]),
        experiment_id=spec["experiment_id"],
        seed=int(spec["seed"]),
        n_seeds=len(seeds),
        checkpoint=_rel(chosen["ckpt"]),
        head_pose_backend=_head_pose_backend(seq_root),
        sequence_root=seq_root,
        deployable=live,
        live_capable=live,
        replay_capable=replay,
        blocked_reason=reason,
        taxonomy=taxonomy,
        ruleset=ruleset,
        objective=objective,
        comparable_group=f"{taxonomy}/{ruleset}/{objective}",
        is_canonical=(taxonomy, ruleset, objective) == CANONICAL_GROUP,
        class_names=classes,
        n_classes=len(classes),
        abstains_on=abstains,
        coverage=float(chosen["val"].get("coverage", 1.0)),
        needs_head_stream="head" in _blocks(fc),
        val=chosen["val"],
        test=chosen["test"],
        val_seed_mean=_seed_mean(seeds, "val"),
        test_seed_mean=_seed_mean(seeds, "test"),
        val_predictions=_rel(chosen["dir"] / "eval_val" / "predictions.npz"),
    )


def _mark_recommended(entries: List[ModelEntry]) -> None:
    """Flag the best model per TAXONOMY, which is what a chooser actually wants.

    34 variants is a research record, not a menu. Most are seeds of an ablation
    whose conclusion is already written down — the feature ladder, the
    architecture sweep, the PRODEN negative result. Offering them all invites
    the comparison the grouping exists to prevent, and buries the four models
    anyone would actually run.

    So the shortlist is one entry per taxonomy, chosen on validation, and:

    * **restricted to the canonical training target** where the taxonomy has one.
      `cue6` has five targets (v1/v2 x single/PRODEN); only v1 single-label is
      the one every published number is measured against, so the v2 and PRODEN
      variants are ablations and stay off the shortlist.
    * **`cue6` gets two entries, deliberately.** The best mean
      (`epochs240/mstcn_556_hp`, [value removed] +- [value removed]) can replay a cached session;
      the best single score (`wave2/mstcn_1074_hp_head`, [value removed]) reads the head
      block and is live-only. Neither dominates, and picking one silently would
      hide a real trade-off.

    Nothing is removed — `recommended` is a display hint. The full registry is
    still scanned, still served by /api/models, and still one checkbox away.
    """
    by_tax: Dict[str, List[ModelEntry]] = {}
    for e in entries:
        if e.live_capable:
            by_tax.setdefault(e.taxonomy, []).append(e)

    for taxonomy, rows in by_tax.items():
        canonical = [e for e in rows
                     if e.ruleset == "v1" and e.objective == "single"]
        pool = canonical or rows
        best = max(pool, key=lambda e: e.val["macro_f1"])
        best.recommended = True
        best.recommended_why = (
            f"best {taxonomy} model on validation"
            + ("" if canonical else " (no canonical-target run exists)"))

        # The live-only/replay-capable split, where it is a real choice rather
        # than a strictly worse option.
        others = [e for e in pool if e is not best
                  and e.replay_capable != best.replay_capable]
        if others:
            alt = max(others, key=lambda e: e.val["macro_f1"])
            alt.recommended = True
            alt.recommended_why = (
                "highest score on this taxonomy, but LIVE-ONLY — reads the head "
                "block, which a session cache does not store"
                if not alt.replay_capable else
                "best that can also re-decide a cached session")


#: Which VLM answers the second opinion, and under which fusion policy.
#: Qwen3-VL-4B-Instruct is ~8 GB in fp16 and is deliberately NOT the 27B that
#: produced the training labels — see the independence caveat in
#: attention/vlm_grounder.py. It weakens the coupling; it does not remove it.
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VLM_POLICY = "agreement"

#: Why the VLM entry is or is not offered, set by the last :func:`_vlm_entries`
#: call and surfaced through /api/models. Without it a missing entry is
#: indistinguishable from a registry that never tried, and the reason lives only
#: in a log line on a host nobody can read.
VLM_STATUS = {"available": False, "reason": "not checked yet",
              "model_id": VLM_MODEL_ID, "transformers": ""}


def _vlm_entries(entries: List[ModelEntry]) -> List[ModelEntry]:
    """One virtual entry per shortlisted cue6 model: that model, plus a VLM.

    Virtual because there is nothing new to train or store. The temporal model
    runs exactly as it does alone; the VLM scores the same detector boxes
    against the six cue phrases, and `attention.fusion` combines the two
    opinions. What changes is the DECISION, not the weights, so the entry
    carries the base model's checkpoint and calibration.

    Restricted to cue6 because `fusion.py` and `vlm_grounder.py` are both built
    on `CUE_CLASSES` — the option letters, the phrase list and the fused vector
    are all six-wide. A cue9 grounder needs its own phrases and is a separate
    piece of work, not a flag.

    Restricted to REPLAY-CAPABLE models for a practical reason: the VLM needs
    the frame, and a live run that also loads an 8 GB VLM per frame is slower
    than the camera path is useful at. The session path already has the pixels.
    """
    # Only offer it where it can actually run. The wiring, the fusion and the
    # calibration fallback are all in place and stub-tested, but Qwen3-VL needs
    # transformers >= 4.45 and this stack pins 4.44.2 deliberately (bumping it
    # risks mmcv's compiled _ext against torch 2.2.2). A dropdown entry that
    # raises on selection is worse than no entry.
    tv = ""
    try:
        import sys
        sys.path.insert(0, str(REPO / "LLMDet"))
        import transformers
        tv = transformers.__version__
        from attention.vlm_grounder import QwenGrounder
        ok, why = QwenGrounder.available()
    except Exception as e:                                   # noqa: BLE001
        ok, why = False, f"{type(e).__name__}: {e}"
    VLM_STATUS.update({"available": bool(ok), "reason": why or "available",
                       "model_id": VLM_MODEL_ID, "transformers": tv})
    if not ok:
        print(f"[registry] VLM entries not offered (transformers {tv}): {why}")
        return []

    def _has_phrases(taxonomy: str) -> bool:
        """Can this taxonomy's classes be presented as an option list?

        A class with no cue phrase is not scorable: the options are a lettered
        list and a gap shifts every letter after it onto the wrong cue, so the
        scores come back well-formed and wrong. `onoff_reliable` and
        `coarse3_reliable` have no phrases (`on_task`, `off_task`,
        `down_or_hidden` are merges, not observable behaviours) and are skipped.
        """
        try:
            import sys
            sys.path.insert(0, str(REPO / "LLMDet"))
            from attention.cue_phrases import phrases_in_class_order
            from attention.taxonomy import taxonomy_classes
            phrases_in_class_order(taxonomy_classes(taxonomy))
            return True
        except Exception:
            return False

    out = []
    for e in entries:
        if not (e.recommended and e.live_capable and e.replay_capable
                and _has_phrases(e.taxonomy)):
            continue
        v = ModelEntry(**{**asdict(e),
                          "variant_id": f"{e.variant_id}+vlm",
                          "label": f"{e.label} · + VLM second opinion",
                          "sweep_label": "temporal model fused with a VLM",
                          "vlm": True,
                          "vlm_base": e.variant_id,
                          "vlm_model_id": VLM_MODEL_ID,
                          "vlm_policy": VLM_POLICY,
                          "is_default": False,
                          # NOT recommended, and that is a measurement rather
                          # than caution. Fusion was evaluated on the val split
                          # (tools/bench_vlm_fusion.py, [internal notes, not included] and does not
                          # help: `agreement` reproduces the temporal model
                          # (measured value removed from the handover copy)
                          # abstaining on 27% of them, and the two policies that
                          # (measured value removed from the handover copy)
                          # (measured value removed from the handover copy)
                          # ablation, not because anyone should select it.
                          "recommended": False,
                          "recommended_why": (
                              f"{e.variant_id} plus a second opinion from "
                              f"{VLM_MODEL_ID} under the '{VLM_POLICY}' policy. "
                              f"MEASURED NOT TO HELP: agreement matches the "
                              f"temporal model on the frames it answers and "
                              f"abstains on 27%; pool -0.072 and product -0.135 "
                              f"macro-F1. Kept as an ablation. Seconds per "
                              f"frame, not frames per second.")})
        out.append(v)
    return out


def scan(thesis_root: Path = THESIS) -> List[ModelEntry]:
    """One entry per variant, best validation seed, ordered best-first.

    Ordering is: live-capable first, then the canonical target
    (``cue6``/v1/single-label) before every other, then by validation macro-F1
    *within* that group. The group term is what stops a 2-class abstaining model
    from outranking a 6-class one on a number that is not comparable to it.
    """
    entries = [_make_entry(vid, seeds, max(seeds, key=lambda r: r["val"]["macro_f1"]))
               for vid, seeds in _collect(thesis_root).items()]
    entries.sort(key=lambda e: (not e.live_capable, not e.is_canonical,
                                e.comparable_group, -e.val["macro_f1"]))
    for e in entries:
        # The default has to work in every mode the dashboard offers, and the
        # dashboard BOOTS into session replay. wave2/mstcn_1074_hp_head scores
        # (measured value removed from the handover copy)
        # which no session cache stores — defaulting to it would land every
        # visitor on a model that cannot re-decide the session in front of
        # them. It stays offered, and live analysis can select it.
        if e.sweep in PROBE_SWEEPS:
            continue
        if e.live_capable and e.replay_capable and e.is_canonical:
            e.is_default = True     # best canonical-target variant on validation
            break
    _mark_recommended(entries)
    entries += _vlm_entries(entries)
    return entries


def default_entry(entries: List[ModelEntry]) -> ModelEntry:
    for e in entries:
        if e.is_default:
            return e
    raise SystemExit("no deployable model on the canonical target in the registry")


def best_in_group(entries: List[ModelEntry], taxonomy: str,
                  ruleset: str = "v1", objective: str = "single",
                  live_only: bool = True) -> Optional[ModelEntry]:
    """Best validation variant for one target — e.g. the best coarse model.

    Exists so callers can ask for "the best onoff_reliable model" without
    re-deriving the comparability rule, and without sorting across targets.
    """
    want = f"{taxonomy}/{ruleset}/{objective}"
    cands = [e for e in entries if e.comparable_group == want
             and (e.live_capable or not live_only)]
    return max(cands, key=lambda e: e.val["macro_f1"]) if cands else None


def find(entries: List[ModelEntry], variant_id: str,
         thesis_root: Path = THESIS) -> Optional[ModelEntry]:
    """Look up a variant, optionally pinned to one seed as `variant@sNN`.

    Representing a variant by its best VALIDATION seed is the right rule for a
    ranked dropdown and the wrong one for a deployment. `attention_runtime.yaml`
    deploys ff_det/mstcn_553_ff_s42, but the variant id `ff_det/
    mstcn_553_facefound` resolves here to s43 -- a different checkpoint carrying
    its own fitted calibration (T [value removed], alert 0.66 against T [value removed], alert
    0.64) and a different alert coverage (64.6% against 69.0%).

    Without a pin the live Space cannot serve the checkpoint the thesis names as
    deployed, so a demo offered as "the deployed system" quietly is not one.
    Pinning changes nothing about how the dropdown ranks or defaults.
    """
    if "@s" not in variant_id:
        return next((e for e in entries if e.variant_id == variant_id), None)

    base, _, seed_txt = variant_id.partition("@s")
    try:
        seed = int(seed_txt)
    except ValueError:
        return None
    seeds = _collect(thesis_root).get(base)
    if not seeds:
        return None
    chosen = next((r for r in seeds if int(r["spec"]["seed"]) == seed), None)
    if chosen is None:
        return None
    entry = _make_entry(base, seeds, chosen)
    # Keep the pin visible: it is what the UI shows, what /api/models reports
    # as active, and what calibration_path() resolves against.
    entry.variant_id = variant_id
    return entry


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write the registry as JSON")
    args = ap.parse_args()

    entries = scan()
    group = None       # (comparable_group, live_capable)
    hdr = (f"{'variant':46} {'best':5} {'val F1':>7} {'val mean+-sd':>16} "
           f"{'test F1':>8} {'cls':>3} {'cov':>5} {'live':5} {'replay':6} sweep")
    for e in entries:
        if (e.comparable_group, e.live_capable) != group:
            group = (e.comparable_group, e.live_capable)
            tag = "  <- canonical target" if e.is_canonical else ""
            if not e.live_capable:
                tag += "  [NOT DEPLOYABLE]"
            print(f"\n=== target {e.comparable_group}{tag} ===")
            print(hdr)
            print("-" * len(hdr))
        test = f"{e.test['macro_f1']:.4f}" if e.test else "—"
        vm = e.val_seed_mean
        mean = f"{vm['macro_f1']:.4f}+-{vm['macro_f1_sd']:.4f}" if vm else "—"
        print(f"{e.variant_id:46} s{e.seed:<4} "
              f"{e.val['macro_f1']:7.4f} {mean:>16} {test:>8} "
              f"{e.n_classes:3d} {e.coverage:5.3f} "
              f"{'yes' if e.live_capable else 'NO':5} "
              f"{'yes' if e.replay_capable else 'NO':6} "
              f"{e.sweep_label}"
              + ("   <- DEFAULT" if e.is_default else ""))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            [e.to_json() for e in entries], indent=2))
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
