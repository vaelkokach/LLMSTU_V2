# FINAL TEST-SPLIT EVALUATION — run exactly once (THESIS_PLAN P0.6).
# Pre-registered in [internal notes, not included] at commit 87bb2db.
# odvg_test.jsonl: 9,405 frames over 27 videos, disjoint from the 73 train and
# 27 val videos, never read before this run.
_base_ = ['./student_llmstu_exact.py']
test_ann = '../llmstu_tools/outputs/odvg_test.jsonl'
val_dataloader = dict(dataset=dict(ann_file=test_ann))
test_dataloader = val_dataloader
