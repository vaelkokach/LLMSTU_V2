# Matching-ablation ARM A: ordinal unit->box assignment (old pipeline
# behavior) over the same frames/captions/boxes as the other arms.
# Short schedule — the arms compare label-assignment quality, not final
# performance. 4-GPU launch: bash dist_train.sh configs/student_ablation_ordinal.py 4 --amp
# LR note: see student_llmstu_exact.py (auto_scale_lr handles global batch 8).

_base_ = ['./grounding_dino_swin_t_student_only.py']

arm_train_ann = '../llmstu_tools/outputs/odvg_ablation_ordinal.jsonl'
arm_val_ann = '../llmstu_tools/outputs/odvg_val.jsonl'

train_dataloader = dict(dataset=dict(ann_file=arm_train_ann))
val_dataloader = dict(dataset=dict(ann_file=arm_val_ann))
test_dataloader = dict(dataset=dict(ann_file=arm_val_ann))

max_iter = 10000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=1000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
         end=500),
    dict(type='MultiStepLR', begin=0, end=max_iter, by_epoch=False,
         milestones=[7000, 9000], gamma=0.1),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        max_keep_ckpts=10))
