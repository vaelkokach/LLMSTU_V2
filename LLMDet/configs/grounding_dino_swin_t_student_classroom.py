# Student classroom: visual grounding for action/emotion phrases (VG mode).
#
# Setup (from LLMDet repo root, sibling layout as official grounding_data):
#   ../grounding_data/stu_img/frames/                    <- image files (filename in jsonl)
#   ../grounding_data/stu_img/annotations/Qwen3-VL_Student_action_emotion_fixed_train.jsonl
#   ../grounding_data/stu_img/annotations/Qwen3-VL_Student_action_emotion_fixed_val.jsonl
# Create train/val (e.g. 80/20) with Qwen3-VL/split_student_jsonl_train_val.py
# Plus COCO / Flickr30k / GQA trees as in grounding_dino_swin_t.py (same paths).
#
# Regenerate or repair labels in Qwen3-VL so tokens_positive are *character* spans:
#   python fix_jsonl_tokens_positive.py --in ... --out ... --validate-bert
#
# Train (8x GPU example):
#   bash dist_train.sh configs/grounding_dino_swin_t_student_classroom.py 8 --amp
#
# Mix ratio: with per-GPU batch_size=2, MultiSourceSampler cannot realize ~80/20 (integer
# split). We use RepeatDataset on the student split so one ConcatDataset epoch is roughly
#   P(general) ≈ (|coco|+|flickr|+|gqa|) / (student_domain_repeat * |student| + |coco| + ...).
# Tune student_domain_repeat upward if your student jsonl is small (more student per cycle),
# downward if you want more general data per cycle (closer to 30% VG).
#
# Use {{ _base_.name }} for symbols defined only in the merged base config (train_pipeline,
# dataset dicts, etc.). Plain `train_pipeline` is not in global scope for this file.

_base_ = ['./grounding_dino_swin_t.py']

# Same official MM-Grounding-DINO Swin-T init as grounding_dino_swin_t.py (explicit).
load_from = (
    '../huggingface/mm_grounding_dino/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')

# Default ODVG cleaner removes clauses with "not", "no", "appear", "seems", etc. — destructive
# for emotion/action text. Student split keeps captions verbatim; COCO/Flickr/GQA use base
# settings (inherited dataset dicts still reference clean_caption=True, use_uniform_prompt=True).
clean_caption = False
use_uniform_prompt = False
# Must exist in this file's global scope (same value as grounding_dino_swin_t.py line 11).
use_short_cap = False

data_root_student = '../grounding_data/stu_img/'
student_train_ann = 'annotations/Qwen3-VL_Student_action_emotion_fixed_train.jsonl'
student_val_ann = 'annotations/Qwen3-VL_Student_action_emotion_fixed_val.jsonl'

student_classroom_train = dict(
    type='ODVGDataset',
    data_root=data_root_student,
    ann_file=student_train_ann,
    data_prefix=dict(img='frames/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline={{_base_.train_pipeline}},
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

# --- Mix ~10–30% COCO+Flickr+GQA (see header): repeat student so domain data is not drowned ---
# Inherited from base (same paths / jsonls as grounding_dino_swin_t.py):
#   coco2017_train_dataset, flickr30k_dataset, gqa_dataset
student_domain_repeat = 100

# Anti-forgetting mix toggle. False = student-only ×repeat (faster, but the model can forget
# general grounding — measure with eval_mode='lvis'). True = concat COCO/Flickr/GQA per the
# header's mix-ratio math. The Mar-2026 50k-iter run used the mix; set False for ablations.
use_general_mix = False

student_classroom_train_repeated = dict(
    type='RepeatDataset',
    times=student_domain_repeat,
    dataset=student_classroom_train)

_general_mix_datasets = [
    {{_base_.coco2017_train_dataset}},
    {{_base_.flickr30k_dataset}},
    {{_base_.gqa_dataset}},
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[student_classroom_train_repeated] +
        (_general_mix_datasets if use_general_mix else [])))

# ---------------------------------------------------------------------------
# Validation/Eval toggles
#   eval_mode='student' -> in-domain classroom grounding (Flickr30kMetric)
#   eval_mode='lvis'    -> forgetting probe on LVIS (LVISMetric bbox AP)
# quick_debug=True limits eval to the first quick_n samples.
# ---------------------------------------------------------------------------
eval_mode = 'student'  # 'student' or 'lvis'
quick_debug = False # True or False
quick_n = 100

student_val_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadTextAnnotations'),
    dict(type='ODVGPhraseEvalPrep'),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'text', 'custom_entities', 'tokens_positive', 'phrases',
            'phrase_ids')),
]

student_classroom_val = dict(
    type='ODVGDataset',
    data_root=data_root_student,
    ann_file=student_val_ann,
    data_prefix=dict(img='frames/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=student_val_pipeline,
    return_classes=True,
    actual_dataset_mode='VG',
    use_short_cap=use_short_cap,
    use_uniform_prompt=use_uniform_prompt,
    clean_caption=clean_caption,
    backend_args=None)

student_val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=student_classroom_val)

student_val_evaluator = dict(
    _delete_=True,
    type='Flickr30kMetric',
    topk=(1, 5, 10, -1),
    iou_thrs=0.5)

data_root = '../grounding_data/coco/'
lvis_dataset = dict(
    data_root=data_root,
    type='LVISV1Dataset',
    ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
    data_prefix=dict(img=''),
    test_mode=True,
    pipeline={{_base_.test_pipeline}},
    return_classes=True)
if quick_debug:
    lvis_dataset['indices'] = list(range(quick_n))

lvis_val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=lvis_dataset)

lvis_val_evaluator = dict(
    _delete_=True,
    type='LVISMetric',
    ann_file=data_root + 'annotations/lvis_v1_minival_inserted_image_name.json',
    metric='bbox')

if eval_mode == 'student':
    val_dataloader = student_val_dataloader
    test_dataloader = student_val_dataloader
    val_evaluator = student_val_evaluator
    test_evaluator = student_val_evaluator
elif eval_mode == 'lvis':
    val_dataloader = lvis_val_dataloader
    test_dataloader = lvis_val_dataloader
    val_evaluator = lvis_val_evaluator
    test_evaluator = lvis_val_evaluator
else:
    raise ValueError(f'Unknown eval_mode: {eval_mode}')

# Shorter schedule than full 150k-it pretrain; bump if you train longer.
max_iter = 50000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=10000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[35000, 45000],
        gamma=0.1)
]

# Do not use _delete_=True here: it drops timer/param_scheduler/sampler_seed and MMEngine
# requires each hook dict to include "type" (e.g. CheckpointHook).
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=10),
    visualization=dict(type='GroundingVisualizationHook'),
    logger=dict(type='LoggerHook', interval=50))
log_processor = dict(by_epoch=False)

# Slightly lower LR when fine-tuning on a narrow domain (tune if loss diverges).
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

# Optional: resume your own run instead of official weights above.
# resume = True
