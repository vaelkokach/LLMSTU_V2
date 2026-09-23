# Guideline

How to work on this project. `Documentation.md` says what the code is; this says what to
do and what not to do.

Everything below was learned the expensive way on the original project — each item cost
somebody weeks. None of it is theory. Read it once now and once again when something
starts behaving oddly.

**Authorship.** Like `Documentation.md` and the entire codebase, this document was
written by Claude (Anthropic's Claude Code). No human wrote the software you are
inheriting.

---

## 1. The rules that protect your results

### Split by recording and by room. Never by clip.

If frames from one recording end up in both training and testing, your numbers are
inflated and your project is worthless — and it will look like it worked, which is why
this is dangerous. The same student in the same chair under the same light appears on
both sides, and the model recognises the room rather than the behaviour.

This is not hypothetical. Published engagement benchmarks do it: on one of them, moving
from the official clip-level split to a subject-disjoint split made agreement with the
ratings collapse. The original project's first round of results had to be thrown away
entirely for the same reason.

For your project the rule is stricter, because your whole claim is about environments:
**no room may appear on both sides of a split.** Leave-one-environment-out means exactly
that — train on rooms A, B, C, test on D, and repeat.

Add a room identifier to every record when you collect it. Recovering it later is
possible but miserable; `recover_video_ids.py` exists because the original project had to.

### Decide what the test set is, in writing, before you look at it.

Write down which rooms and recordings are the test set, and what you will report from
them, and commit that file. Then evaluate on it **once**, at the end. Every intermediate
decision — which features, which model, which window — comes from the validation split.

If you evaluate on the test set repeatedly and pick the best, you have tuned on it, and
your report says something that is not true.

### A result with one seed is not a result.

Train three seeds and report mean and standard deviation. Compare two configurations
pair-wise, seed against seed, and say in how many pairs the difference held. If a change
helps in one seed of three, it did not help.

Frames within a recording are strongly correlated — a cue persists for tens of seconds.
Treating them as independent observations makes confidence intervals about ten times too
narrow. Bootstrap at the **video** level, not the frame level.

### Check that the training budget is not what you are measuring.

If your best checkpoint keeps landing in the last few epochs while the loss is still
falling, you are measuring the budget, not the method. Raise the budget and re-run. On
the original project, tripling the epoch budget moved every label space upward and
dissolved an "instability" that had been attributed to the taxonomy.

---

## 2. The rules that protect your labels

### Annotate the schema, never the cue.

Fill in the ten fields. Let `taxonomy.py` compute the cue. If you label cues directly,
you can never change the rules without re-annotating everything, and you lose the ability
to test a second taxonomy for free.

### Pre-filling an annotation form anchors the annotator.

The tool shows the VLM's answer and lets a human correct it. That makes annotation fast
and it biases the human hard. On the original project, one annotator accepted the
pre-filled answer far more readily than a second one did, and the measured "agreement
between the model and humans" changed substantially depending on which of the two you
asked. Same crops, same model.

So: the agreement between your pseudo-labels and a human who saw the pre-filled answer is
an **upper bound**, not a measurement. The number that means something is human-to-human
agreement on a subset labelled independently. Measure it (`make_iaa_manifest.py`,
`compute_agreement.py`) and report it.

### Interpretive fields are much weaker than objective ones.

Whether a phone is visible, whether the person is talking, whether they are occluded, what
the hands are doing — two people agree on these almost perfectly, and the answer barely
changes when you reword the prompt. Gaze direction, attention target and engagement level
are far less stable: they move with the prompt wording and the annotator.

Design around that. Build your claims on the objective fields. If you want to report on an
interpretive one, report its agreement alongside it, every time.

### Measure prompt sensitivity before you trust a VLM label.

Write three or four wordings of the prompt, run them on a small tuning subset, and count
how often each field gets the same answer under all of them. Do this **before** you label
the whole corpus. It takes an afternoon and tells you which fields will be learnable.

Also rotate the order of the options when you score a multiple-choice question. If the
model's preference follows the letter rather than the class, it is not answering about the
image.

---

## 3. The rules that protect your pipeline

### Bind descriptions to students by content, never by order.

When a model describes several students in one frame and you need to know which box each
description belongs to, do not assume the order of the descriptions matches the order of
the detections. It does not. Ordinal binding is wrong far more often than it is right, so most of the
training labels end up supervising the wrong student — and it gets worse as the room
fills, because one missed detection shifts every binding after it.

Use an assignment: build a cost matrix between descriptions and boxes from an image–text
similarity, and solve it with the Hungarian algorithm. `LLMDet/matching/` has the code.

### Deduplicate frames, not students.

At one frame per second, a still student produces long runs of nearly identical crops.
Removing them is right. But when you select a frame, restore **every** student in it —
otherwise the students you did not select become unlabelled background and teach the
detector to ignore real people.

### Serve the model at the rate it was trained at.

A model trained on one sample per second, fed every frame of a 30 fps video, sees a
window covering one second of real time instead of thirty. The original project shipped
this bug and it cost more accuracy than every modelling improvement in the thesis
combined. Admit a frame into a student's history only when one second of **source** time
has passed — source time, so a recording replayed at four times speed behaves the same.

### Fail loudly.

Every serious defect on the original project produced a plausible-looking output rather
than an error:

- a dashboard served cues from a randomly initialised network for weeks, because a size
  mismatch was swallowed by a broad `except`;
- feature vectors shorter than expected were zero-padded, making a missing measurement
  indistinguishable from a real one;
- a nine-option question was scored against six offered options, and a softmax over
  nine arbitrary logits is still a valid distribution;
- session analysis silently stopped after 900 frames, so the demonstration covered
  less than half its video.

So: no bare `except` around model loading. No zero-padding a feature vector. Assert the
width. Derive configuration from the checkpoint itself rather than a separate file that
can drift. Test the property that matters — that the options offered equal the options
scored — not that the code runs.

### When a number surprises you, suspect the measurement first.

Twice on the original project a spectacular result turned out to be a benchmark bug: a
spatial prior that leaked the ground-truth ordering into the matching cost, and a window
comparison where longer windows were scored on an easier set of positions. Both looked
like breakthroughs. Before you report a surprising number, try to break it.

---

## 4. Working on the HPC

Read the HPC section of `Documentation.md` for the mechanics. This is the etiquette, and
it matters because the box is shared.

- **Never occupy more than half the GPUs.** The working rule on this box is at most four
  of the eight at once. Check `nvidia-smi` before you launch anything.
- **Ask before any training job.** Explicit approval from your supervisor, every time.
  A sweep left running over a weekend blocks everyone else's work.
- **Do not change the shared environment.** `transformers` is pinned at 4.44.2 and that
  pin protects mmcv against torch 2.2.2. If you need something newer, make a throwaway
  venv with `--system-site-packages` — see `tools/serve_hpc.py`.
- **PyPI is unreachable.** You cannot install packages. Vendor what you need into the
  repository so it arrives over GitHub, or do that piece of work on Colab.
- **Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`** so the cached weights under
  `~/.cache/huggingface` are used and nothing tries to phone home mid-run.
- **Keep your data in your own work directory**, readable only by your group. Never
  commit a recording, a crop or an annotation file. `.gitignore` already blocks
  `grounding_data/` and `huggingface/`; do not add exceptions.
- **Write down what you ran.** Which config, which commit, which GPUs, which seed. A
  result you cannot re-run is not a result, and three months from now you will not
  remember.

---

## 5. Before you record anything

This is the part that can invalidate the whole project after the fact, and the only part
with no technical fix.

- Ethics approval, or a documented exemption, for every room you record in. Obtain it in
  writing before the first recording, not afterwards.
- Informed consent from every student who appears, covering what you record, what you
  keep, who sees it, how long you hold it and what happens at the end.
- A storage plan: who has access, where it lives, when it is deleted.
- Anonymisation for anything that leaves the group. Every face in every image in this
  repository is pixelated; hold yourself to the same line in your report, your slides and
  your defence.
- Agree in advance with your supervisor who owns the resulting dataset and who is named
  on anything published from it.

Describe cues, never people. The system reports that a head is down; it does not report
that a student is bored, inattentive or lazy. Keep that discipline in your writing as
well as your code — a sentence claiming to detect attention is a claim you cannot support
and a reviewer will ask about it.

---

## 6. Making your dataset merge with the others

Two groups are extending this system, and the original corpus already exists. If all
three use the same shapes, the combined dataset is worth far more than three separate
ones. So:

- the same ten fields with the same allowed values — `tools/gold_annotator/vocab.py`,
  unchanged;
- the same precedence rules — `LLMDet/attention/taxonomy.py`, unchanged;
- one JSONL row per student per frame, in the shape of
  `format_sample/sample_labels.jsonl`;
- a stable identifier per recording, per room and per seat on every row;
- your splits stored as a file, not reconstructed by a script from filenames.

If you need a new field — and for multi-environment work you will, at minimum a room
identifier — **add** it. Do not repurpose an existing one, and do not change an allowed
value. Tell the other group what you added.
