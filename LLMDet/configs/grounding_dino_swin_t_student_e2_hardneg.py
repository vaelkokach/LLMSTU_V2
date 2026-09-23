_base_ = ['grounding_dino_swin_t_student_e1.py']

# Thesis E2: student + hard-negative continuation fine-tuning.
# Create `../grounding_data/stu_img/student_hard_negative_vg7_train.jsonl`
# with classroom object negatives (desk, monitor, chair, keyboard, phone).

hard_negative_dataset = dict(
    type='ODVGDataset',
    data_root='../grounding_data/stu_img/',
    ann_file='student_hard_negative_vg7_train.jsonl',
    label_map_file=None,
    data_prefix=dict(img='frames/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.student_behavior_dataset.pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=False,
    use_uniform_prompt=True,
    clean_caption=True,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            _base_.student_behavior_dataset,
            hard_negative_dataset,
        ]))

max_iter = 15000
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=max_iter, val_interval=1000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=max_iter, by_epoch=False, milestones=[10000, 13000], gamma=0.1),
]

load_from = 'work_dirs/grounding_dino_swin_t/iter_15000.pth'
