# P1.1 — ZERO-SHOT cross-dataset evaluation on SCB-Dataset.
#
# Purpose: show that the LLMSTU-trained grounding detector transfers to an
# UNSEEN classroom type with NO fine-tuning. LLMSTU is a computer lab (students
# facing monitors, wide 2812x1050 shot); SCB is a lecture hall (hand-raising,
# writing at desks, facing a teacher). The domain gap is the experiment, not a
# defect.
#
# NOTHING is trained here. The checkpoint is frozen; this is test-only.
#
# Phrases in the SCB ODVG are copied VERBATIM from regen_odvg.ACTIVITY_PHRASE
# ("sleeping head down", "talking to peer") so the domain gap is not confounded
# with a vocabulary gap — the model sees exactly the text it was trained on.
#
# Run:
#   bash dist_test.sh configs/eval_scb_zeroshot.py \
#       work_dirs/thesis_bundle/checkpoints/main_llmstu_exact_iter25000_final.pth 1

_base_ = ['./student_llmstu_exact.py']

# SCB subset root; ODVG filenames are relative to it ("images/val/xxx.jpg").
scb_root = ('../grounding_data/external/SCB/SCB_BowTurnHead_20250509/'
            'SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17/')
# Copied into scb_root so the path is unambiguous (ann_file resolves
# relative to data_root, and the ../.. arithmetic was error-prone).
# Source: grounding_data/llmstu_tools/outputs/odvg_scb_bowturnhead_val.jsonl
scb_ann = 'odvg_scb_bowturnhead_val.jsonl'

val_dataloader = dict(
    dataset=dict(
        data_root=scb_root,
        ann_file=scb_ann,
        data_prefix=dict(img='')))
test_dataloader = val_dataloader
