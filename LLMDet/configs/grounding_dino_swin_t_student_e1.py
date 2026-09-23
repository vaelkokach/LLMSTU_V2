_base_ = ['grounding_dino_swin_t.py']

# Thesis E1: student-only continuation fine-tuning from iter_15000.

student_behavior_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/stu_img/',
    ann_file='student_behavior_emotion_vg7_train.jsonl',
    label_map_file=None,
    data_prefix=dict(img='frames/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=_base_.use_short_cap,
    use_uniform_prompt=_base_.use_uniform_prompt,
    clean_caption=_base_.clean_caption,
    backend_args=None)

student_behavior_val_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/stu_img/',
    ann_file='student_behavior_emotion_vg7_val.jsonl',
    label_map_file=None,
    data_prefix=dict(img='frames/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.test_pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=_base_.use_short_cap,
    use_uniform_prompt=_base_.use_uniform_prompt,
    clean_caption=_base_.clean_caption,
    backend_args=None,
    test_mode=True)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type='ConcatDataset', datasets=[student_behavior_dataset]))

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=student_behavior_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='ODVGRecallMetric',
    iou_thrs=0.5,
    topk=(1, 5, 10))
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

max_iter = 12000
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=max_iter, val_interval=1000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_iter, by_epoch=False, milestones=[8000, 10000], gamma=0.1),
]
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, max_keep_ckpts=20),
    visualization=dict(type='GroundingVisualizationHook'),
    logger=dict(type='LoggerHook', interval=50))

load_from = 'work_dirs/grounding_dino_swin_t/iter_15000.pth'
