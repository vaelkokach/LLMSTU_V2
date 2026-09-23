# DEPLOYMENT override of student_llmstu_exact.py: identical detector, no LMM.
#
# This changes what is CONSTRUCTED, not what is PREDICTED. The LMM is training
# machinery only:
#
#   * every use of `self.lmm` in mmdet/models/detectors/grounding_dino.py sits
#     inside `loss()` (lines 958-1169), which computes loss_lmm_region and
#     loss_lmm_image;
#   * `predict()` (line 1172 onward) — the whole inference path — contains no
#     reference to `self.lmm` at all;
#   * `__init__` deletes the vision tower straight after loading it
#     (grounding_dino.py:449), so SigLIP's 3.5 GB checkpoint is read and
#     immediately discarded even during training.
#
# So on the deployed path the LMM costs a 0.5B-parameter model plus a 3.5 GB
# SigLIP read, and contributes nothing to a single box or score. On a 15 GB
# Space that was enough to push the container into swap: the detector never
# finished loading and even trivial HTTP requests stopped answering.
#
# Detection outputs under this config are identical to student_llmstu_exact.py
# by construction, not by measurement — the code path that produces them does
# not read the module this file removes.
#
# The checkpoint still carries lmm.* and vision_projector.* tensors. mmdet's
# load_checkpoint is non-strict, so those are reported as unexpected keys and
# ignored. That is expected here, not a warning to chase.
#
# attention_runtime.yaml is deliberately NOT edited to point here: it is the
# configuration the thesis cites, and it stays byte-identical. The live Space
# selects this file through the DETECTOR_CONFIG environment variable
# (deploy/hf_space_live/app.py), so the override is a property of the
# deployment, not of the recorded experiment.

_base_ = ['./student_llmstu_exact.py']

model = dict(lmm=None)
