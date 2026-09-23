# DEPLOYMENT override of grounding_dino_swin_t_original.py: identical detector,
# no LMM. This is the OBJECT detector -- the pretrained MM-GroundingDINO that
# answers "cell phone" and "laptop" for the six v1080 object columns, not the
# fine-tuned student detector.
#
# The argument is the same one student_llmstu_exact_deploy.py makes, and it is
# an argument about the code path rather than a measurement: every use of
# `self.lmm` in mmdet/models/detectors/grounding_dino.py is inside `loss()`;
# `predict()` never reads it. Constructing it therefore costs a 0.5B model and
# a 3.5 GB SigLIP read that is discarded immediately, and changes no box and no
# score.
#
# It matters more here than for the student detector. The object columns were
# PRECOMPUTED over 283,913 crops with the LMM present
# (attention/precompute_objects.py), and the live path must produce the same six
# numbers from the same frame -- a checkpoint trained on cached columns and
# served from differently-produced ones is the silent-nonsense failure the named
# LAYOUTS exist to prevent. Because predict() does not read the LMM, the two are
# the same function; `attention/tests/test_object_features.py` pins the claim on
# real frames so it is measured as well as argued.
#
# Unlike the student detector this is reached through the runtime YAML's
# `object_detector.config_path` rather than an environment variable: there is no
# published experiment citing an object-detector config to keep byte-identical,
# and a second override mechanism for the same reason would be one too many.

_base_ = ['./grounding_dino_swin_t_original.py']

model = dict(lmm=None)
