# MAIN thesis detector run: exact-correspondence LLMSTU ODVG, leak-free
# video-wise split. Derived from the best-known recipe (student_only:
# unfrozen backbone, lr_mult 0.25/0.15, AMP).
#
# 4-GPU launch (hard cap, never 8):
#   bash dist_train.sh configs/student_llmstu_exact.py 4 --amp
# LR: optimizer lr stays 8e-5; auto_scale_lr (base_batch_size=16, enable=True,
# inherited from grounding_dino_swin_t.py) scales it by 8/16 -> 4e-5 at the
# 4-GPU global batch of 8. Do NOT also halve it by hand.
#
# Checkpoints: max_keep_ckpts=16 keeps EVERY 2500-iter checkpoint of the run.
# After the run, copy the best-R@1 checkpoint into work_dirs/thesis_bundle/
# BEFORE any cleanup — the March runs lost their best checkpoints to
# max_keep_ckpts pruning.

_base_ = ['./grounding_dino_swin_t_student_only.py']

# data_root stays '../grounding_data/stu_img/' (images under frames/);
# annotations live in llmstu_tools/outputs, reached relative to data_root.
llmstu_train_ann = '../llmstu_tools/outputs/odvg_train.jsonl'
llmstu_val_ann = '../llmstu_tools/outputs/odvg_val.jsonl'

train_dataloader = dict(dataset=dict(ann_file=llmstu_train_ann))
val_dataloader = dict(dataset=dict(ann_file=llmstu_val_ann))
test_dataloader = dict(dataset=dict(ann_file=llmstu_val_ann))

# --- Schedule: complete 25k run, resumable ---------------------------------
# 25000 iters at global batch 8 over 36,339 images = 5.5 epochs.
#
# WHY 25k AND NOT 40k: the 40k figure was inherited from the March
# student_only config, where R@1 climbed all the way to iter 37500 on a
# TEMPORALLY LEAKED split — i.e. the apparent gain past 25k is consistent with
# memorising near-duplicate val frames, not generalisation. On the clean
# video-wise split, ARM B converged in 3.3 epochs (10k iters / 24,218 images;
# (measured value removed from the handover copy)
# that, so 40k has no evidential support on leak-free data.
#
# Milestones are at 70% / 90% of the horizon — the same proportions the March
# schedule used (28000/36000 of 40000) — so the LR decays INSIDE this run and
# the final checkpoint is converged and directly citable.
#
# RESUMING: `max_iter` and `sched_horizon` are kept as separate names on
# purpose. To extend the run, raise BOTH and relaunch with --resume; changing
# max_iter alone would leave the LR floored at 4e-7 for the extension. To
# resume an interrupted run unchanged, just relaunch with --resume:
#   bash dist_train.sh configs/student_llmstu_exact.py 4 --amp --resume
# `--resume` (no argument) sets cfg.resume=True and cfg.load_from=None, so the
# runner auto-finds the latest checkpoint and restores optimizer, AMP-scaler,
# scheduler and iteration state. Do NOT pass --resume on the first launch.
max_iter = 25000
sched_horizon = 25000

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=2500)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=1000),
    dict(type='MultiStepLR', begin=0, end=sched_horizon, by_epoch=False,
         milestones=[17500, 22500], gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        max_keep_ckpts=16))
