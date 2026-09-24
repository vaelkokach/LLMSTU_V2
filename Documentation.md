# Documentation

Reference for the code in this repository: what each part does, where everything is
placed, and how to run it — locally and on the HPC.

Read `guideline.md` next. It is short, and it is the part that will save you months.

---

## Authorship

**Every line of code in this repository, and both of its documents, were written by
Claude — Anthropic's Claude Code — over the course of the master's project it comes
from.** No part of the software was written by a human. Claude designed the pipeline,
wrote the detector fine-tuning and temporal-model code, the labelling pipeline, the
annotation tool, the curation tools, the dashboard and the tests, and wrote this
document and `guideline.md`.

What the human supervisor of that project (Vael Kokach) contributed was the direction,
the recordings, the human annotation, and every decision about what was worth
measuring. That division is worth knowing for two reasons: it tells you the code was
produced by a model and should be read rather than trusted, and it tells you the same
approach is open to you.

Third-party code keeps its own licence. `LLMDet/` is a vendored fork of LLMDet /
MM-Grounding-DINO and is Apache 2.0 — see `LLMDet/LICENSE`. Everything else is covered
by `LICENSE` at the root of this repository.

---

## What this repository is, and what it is not

It is the complete software of a system that watches a wide classroom camera view,
finds every student, follows each one over time, and reports a **visible behaviour cue**
per student per second: `screen_oriented`, `looking_away`, `head_down`, `turned_to_peer`,
`phone_use`, or `uncertain`. The word *attention* never means a mental state here; it
means a visible, sustained, off-task cue. Keep that distinction in your report.

**It contains no data and no results.** No recordings, no annotations, no trained
checkpoints, no measured numbers. That is deliberate: the recordings are identifiable
video of real students collected under a consent agreement that does not extend to you,
and the results belong to a thesis that is not yet examined. You are collecting your own
data anyway — that is the project.

What you do get, so you can build against the right shapes, is `format_sample/`:
24 real crops with every face pixelated, plus `sample_labels.jsonl`, one row per crop
with all ten schema fields and both derived cue labels. Match that format exactly and
your dataset will merge with the original corpus and with the other group's.

---

## Where everything is placed

```
Documentation.md          this file
guideline.md              how to work: method rules and HPC etiquette
LICENSE
format_sample/            24 pixelated crops + sample_labels.jsonl
bootstrap.py              resolves and verifies model weights per profile
artifacts.lock.json       what each profile needs, with sizes, hashes and licences
pyproject.toml            the package and its optional dependency sets

pipeline/                 STAGE 1-2: crop students, pseudo-label them with a VLM
  llmstu/                   the package: crop, caption, schema, dedup, quality, sampling
  scripts/                  01_crop.py … 20_finalize_local.py, run in order
  labeling/index.html       the early browser labelling page
  notebooks/                Colab notebooks that drive the same package
  config.yaml               model, prompt, batch and dataset settings

tools/gold_annotator/     STAGE 3: the human verification tool
  serve.py                  local web server for the annotator
  index.html                the annotation UI
  vocab.py                  THE SCHEMA. Ten fields and their allowed values
  make_sample_manifest.py   choose a stratified subset to verify by hand
  make_iaa_manifest.py      choose the subset two people label independently
  compute_agreement.py      Cohen's kappa per field, and on the derived cue
  make_offline_bundle.py    pack crops + manifest for annotating without a server
  derive_gold_events.py     turn verified frames into episodes
  finalize_gold.py          freeze a verified set
  measure_ceiling.py        how well the pseudo-labels agree with the humans

grounding_data/llmstu_tools/   STAGE 4: curation and splits
  build_seat_tracks.py      associate a student's crops over time into a seat track
  dedup_subsample.py        drop near-duplicate frames, cap the majority class
  recover_video_ids.py      recover which recording a frame came from, by appearance
  make_splits.py            video-wise and room-wise splits
  regen_odvg.py             write the detector's training format
  strip_labels.py           produce a label-free copy
  sample_gold_candidates.py draw the stratified verification sample

LLMDet/                   STAGE 5: the models
  attention/                the temporal half — everything after detection
    taxonomy.py               schema -> cue classes. The precedence rules live here
    features.py               the per-student feature vector, block by block
    sequence_builder.py       per-student sequences from tracked records
    temporal_model.py         MS-TCN, ASRF-style and transformer
    tracking.py               tracker, and track_handoff.py for identity changes
    head_pose.py              face backends and the face-found flag
    object_features.py        second detector pass for phones and laptops
    events.py                 cue timeline -> sustained episodes (hysteresis)
    display_smoothing.py      what the UI shows, separate from what the model predicts
    realtime_infer.py         the live path
    thesis_eval/              training, evaluation, calibration, metrics
      train.py                  train one temporal model
      launch_sweep.py           launch a family of runs
      data.py                   feature layouts and configs. Read this before training
      calibrate.py              temperature scaling and threshold selection
      metrics.py, segmentation.py, eval_events.py
    tests/                    the test suite. Keep it green
  configs/                  detector and temporal configs
  mmdet/, llava/, ram/      vendored third-party detector (Apache 2.0)
  tools/                    detector training and evaluation entry points

tools/dashboard/          the instructor-facing web app and the model registry
tools/serve_hpc.py        serve the dashboard from inside the HPC container
deploy/hf_space*/         the Hugging Face Space deployment of the same app
tools/bench_*.py          measurement scripts (frame rate, history length, VLM, FPS)
ci/                       the continuous-integration workflow
```

Two directory names are in `.gitignore` on purpose and will never appear here:
`grounding_data/LLMSTU/` (crops) and `huggingface/` (weights). They are where data and
weights go on your own machine.

---

## The four label layers

This is the single most important thing to understand, and the most common thing to get
wrong. **Cues are never annotated directly.**

| layer | what it is | defined in |
|---|---|---|
| 0 | the student box the detector predicts | `LLMDet/configs/attention_runtime.yaml` |
| 1 | the ten-field annotation schema — the actual labels | `tools/gold_annotator/vocab.py` |
| 2 | six cue classes, derived from layer 1 by fixed rules | `LLMDet/attention/taxonomy.py` |
| 2b | nine cue classes, a second projection of the same schema | `LLMDet/attention/taxonomy.py` |
| 3 | regrouped coarser taxonomies | `LLMDet/attention/taxonomy.py` |

A human — or the VLM — fills in layer 1 only: activity, gaze direction, attention target,
engagement level, posture, hand state, and four booleans (phone visible, laptop visible,
talking, occluded). `taxonomy.py` then computes the cue with a **precedence rule**: the
first matching condition wins, in a fixed order. Change the rules and every label in your
dataset changes, with no re-annotation. That is the whole point of the design, and it is
why you must annotate the schema and not the cue.

`format_sample/sample_labels.jsonl` shows one row per crop in exactly this shape.

---

## Running it

### Stage 1–2 — crop and pseudo-label

```bash
cd pipeline
pip install -r requirements.txt
python scripts/01_crop.py            # detect and crop every student from every frame
python scripts/02_caption.py         # VLM fills the ten schema fields per crop
python scripts/09_dedup.py           # drop near-identical crops
```

`config.yaml` sets the model, the prompt variant, batch size and the dataset names.
Every `CHANGE_ME/...` value in it and in `pipeline/llmstu/config.py` is a placeholder
for **your** dataset repository. Change them before your first upload; if you leave
them, the upload stage fails loudly rather than writing somewhere unintended.

The prompt matters more than you expect. `pipeline/llmstu/prompts.py` holds the wording,
including the guard clause that tells the model to look only at the student in the centre
of the crop, to answer `unknown` when something is not visible, and never to guess
identity, age, gender or ethnicity. Keep that clause.

### Stage 3 — human verification

```bash
python tools/gold_annotator/make_sample_manifest.py   # stratified subset
python tools/gold_annotator/serve.py                  # opens the annotation UI
```

The UI pre-fills every field with the VLM's answer. That is fast and it is dangerous —
see `guideline.md`, the section on anchoring.

For the agreement subset, `make_iaa_manifest.py` then `compute_agreement.py`.

### Stage 4 — curation and splits

```bash
python grounding_data/llmstu_tools/build_seat_tracks.py
python grounding_data/llmstu_tools/dedup_subsample.py
python grounding_data/llmstu_tools/make_splits.py
```

`make_splits.py` splits by recording. For your project you need it to split **by room**
as well — that is a change you will have to make, and it is the heart of your
leave-one-environment-out evaluation. Add a room identifier to every record at collection
time, not afterwards.

### Stage 5 — training

The temporal package imports itself as `attention.*`, so **`LLMDet/` is the import root**,
not the repository root. Run it from inside that directory:

```bash
cd LLMDet
python -m attention.thesis_eval.train --help
python -m attention.thesis_eval.launch_sweep --help
```

From anywhere else, `PYTHONPATH=/path/to/repo/LLMDet` does the same job. The detector's
own entry points (`LLMDet/tools/`, `LLMDet/train.py`) are run from `LLMDet/` too.

Read `LLMDet/attention/thesis_eval/data.py` first. It has two tables: `LAYOUTS`, which
records what a built sequence file physically contains, and `FEATURE_CONFIGS`, which
names the blocks a model reads. A config that asks for a block its layout does not have
is refused rather than zero-padded. Do not "fix" that by padding.

---

## The HPC

You will need an account and the project directory from your supervisor. Fill these in
before you start; they are the only things this document cannot tell you.

```
host                 [HPC HOSTNAME]
your login           [USERNAME]
project directory    [/path/to/project]
shared read-only     [/path/to/shared]      weights, environment, job templates
your work directory  [/path/to/project]/[USERNAME]
Jupyter URL          [https://.../]
```

### What the box is

Eight A100 GPUs, inside a Docker container. Python 3.11, torch 2.2.2+cu121.

**GitHub and huggingface.co are reachable. PyPI is not.** You cannot `pip install`
anything there. This is the single fact that trips up everyone on their first day.
Plan for it: the environment you get is the environment you have. If you need a package
that is not installed, you either vendor it into the repository (so it arrives over
GitHub) or you do that work on Colab instead.

Model weights are already cached under `~/.cache/huggingface`. Use them offline:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### The version pins

`transformers` is pinned at **4.44.2**. That pin is not cosmetic: it protects mmcv's
compiled `_ext` against torch 2.2.2, and moving it breaks the detector for everyone
sharing the box. If you need a newer `transformers` — some vision-language models want
4.45 or later — create a throwaway virtual environment with `--system-site-packages`,
so it inherits torch, mmcv and mmdet unchanged and overrides `transformers` alone.
`tools/serve_hpc.py` does exactly this and is worth reading as the template.

### Reaching a web interface

Only one port is published from the container: **8888**, the Jupyter server. A new port
cannot be opened, because `docker run -p` happens at container creation. The route out is
`jupyter-server-proxy`, which is installed:

```
<your Jupyter URL>/proxy/<port>/
```

`python tools/serve_hpc.py` starts the dashboard, picks a free GPU and prints the proxied
URL. `--port` and `--device` override it, `--stop` shuts it down.

A live camera needs a secure context: `getUserMedia` is silently refused on plain HTTP
unless the origin is localhost. Either reach Jupyter over HTTPS, or tunnel it with
`ssh -L 8888:localhost:8888 [USERNAME]@[HPC HOSTNAME]` so the origin becomes localhost.
Uploading or replaying a recording works either way.

### Getting this repository onto the HPC

Work in **your own clone of this repository**, in your own directory on the box:

```bash
cd [/path/to/project]/[USERNAME]
git clone https://github.com/vaelkokach/LLMSTU_V2.git
cd LLMSTU_V2
```

GitHub is reachable from the container, so this works directly. Do not work inside anyone
else's checkout — the original project's copy on that box contains its data and its
unpublished results, and it is not yours to read or to modify.

The clone you get is exactly the tree mapped out above, and the map stays accurate. What
changes is that three directories appear as you work, none of which are in git and none of
which should ever be committed:

| appears when | what it is |
|---|---|
| `grounding_data/LLMSTU/crops/` | your crops, written by the pipeline, in shard subdirectories |
| `huggingface/` | model weights, staged by `bootstrap.py` |
| `LLMDet/work_dirs/` | training runs: checkpoints, logs and metrics, one directory per run |

If a path in this document does not exist in your clone, it is one of those three, and it
exists once you have run the stage that creates it.

### Getting the weights

```bash
python bootstrap.py --profile research --dry-run   # what it would fetch, and why
python bootstrap.py --profile research
```

Profiles are `demo`, `research`, `full-data` and `hpc`, in increasing order of what they
need. `research` is the one that lets you run the whole detect → track → feature →
classify chain **on your own video** with public weights. `--offline` and `--no-download`
suppress all network access and report gaps instead of fetching.

Artifacts marked `restricted` in `artifacts.lock.json` are never fetched automatically,
with or without network. Those are the original project's data and fine-tuned detector.
The script prints why and stops. That is correct behaviour, not a bug to work around.

### Where your data goes

Not in git. `grounding_data/` and `huggingface/` are ignored, and your recordings must
never be committed — they are identifiable video of people. Keep them in your own HPC
work directory with permissions that only your group can read, and agree the storage and
deletion plan with your supervisor before you record anything.

---

## Things to fill in before you start

- [ ] HPC hostname, username, project directory, shared directory, Jupyter URL (above)
- [ ] Your own Hugging Face dataset names, replacing every `CHANGE_ME/...`
- [ ] Ethics approval and consent forms for the rooms you will record
- [ ] A room identifier field, added to the schema at collection time
