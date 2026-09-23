# Fine-tune MM-Grounding-DINO on *student classroom data only* (no COCO/Flickr/GQA).
#
# Prerequisite — fix text–box alignment in the JSONL (critical if training "learned nothing"):
#   Many Qwen exports stored word-like indices in tokens_positive instead of *character*
#   spans. Recompute spans from each region's phrase text:
#     cd ../Qwen3-VL
#     python fix_jsonl_tokens_positive.py --in Qwen3-VL_Student_action_emotion.jsonl \\
#         --out Qwen3-VL_Student_action_emotion_fixed.jsonl --validate-bert
#   Then split train/val and point ann_file paths below to the *_fixed* files.
#

data_root_student = '../grounding_data/stu_img/'
student_train_ann = 'annotations/Qwen3-VL_Student_action_emotion_fixed_train.jsonl'
student_val_ann = 'annotations/Qwen3-VL_Student_action_emotion_fixed_val.jsonl'
#
# Train (example 8 GPUs):
#   bash dist_train.sh configs/grounding_dino_swin_t_student_only.py 8 --amp

_base_ = ['./grounding_dino_swin_t_student_classroom.py']

# RandomSamplingNegPos2 below needs real Python names. MMEngine does not put
# grand-base symbols (from grounding_dino_swin_t.py) in this file's scope.
# Keep these in sync with grounding_dino_swin_t.py.
lang_model_name = '../huggingface/bert-base-uncased/'
lmm_path = '../huggingface/my_llava-onevision-qwen2-0.5b-ov-2/'
lmm_max_token_length = 1600
num_region_caption = 16

# ODVGDataset kwargs — same as grounding_dino_swin_t_student_classroom.py (not in scope
# from _base_ during this file's execution).
use_short_cap = False
use_uniform_prompt = False
clean_caption = False


# Official MM-Grounding-DINO Swin-T (same init as other student configs).
load_from = (
    '../huggingface/mm_grounding_dino/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')

# Drop images with no boxes (malformed captions / failed phrase extraction).
filter_empty_gt = True


# Unfrozen Swin + backbone gradient checkpoint (with_cp=True) under DDP often hits
# RuntimeError: Expected to mark a variable ready only once. Turn off backbone
# checkpointing here (more VRAM; stable backward). If OOM, set freeze_backbone=True.
model = dict(freeze_backbone=False, backbone=dict(with_cp=False))

student_classroom_train = dict(
    type='ODVGDataset',
    data_root=data_root_student,
    ann_file=student_train_ann,
    data_prefix=dict(img='frames/'),
    filter_cfg=dict(filter_empty_gt=filter_empty_gt),
    pipeline=[
        dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomFlip', prob=0.5),
        dict(
            type='RandomChoiceResize',
            scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                    (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                    (736, 1333), (768, 1333), (800, 1333)],
            keep_ratio=True),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
        dict(
            type='RandomSamplingNegPos2',
            tokenizer_name=lang_model_name,
            tokenizer_name2=lmm_path,
            lmm_max_token_length=lmm_max_token_length,
            num_region_caption=num_region_caption,
            num_sample_negative=85,
            max_tokens=256),
        dict(
            type='PackDetInputs',
            meta_keys=(
                'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                'flip', 'flip_direction', 'text', 'tags', 'contrast_conv',
                'custom_entities', 'tokens_positive', 'dataset_mode',
                'conversations', 'region_conversations')),
    ],
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=student_classroom_train)

# Student-only: shorter cycle, more frequent validation, slightly higher LR.
max_iter = 40000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=2500)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[28000, 36000],
        gamma=0.1)
]

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=8e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.25),
            'language_model': dict(lr_mult=0.15),
        }))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2500,
        max_keep_ckpts=8),
    visualization=dict(type='GroundingVisualizationHook'),
    logger=dict(type='LoggerHook', interval=50))
