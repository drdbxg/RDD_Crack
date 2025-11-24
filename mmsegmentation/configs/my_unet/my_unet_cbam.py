# =========================================================
#   MMseg 1.x 配置
# =========================================================

default_scope = 'mmseg'
randomness = dict(seed=42)

# --------------------- 模型 ---------------------
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255
        # ⚠ 注意：这里没有 size，也没有 size_divisor
    ),

backbone=dict(
        type='MyUNet',
        out_channels=64
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        channels=64,
        num_convs=2,
        num_classes=2,
        dropout_ratio=0.1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        in_index=-1,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    )
)

# ------------------- 数据集 ---------------------
dataset_type = 'CrackDataset'
data_root = 'data/UAVCrack'
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# ----------------- 优化器 -----------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4)
)

# ---------------- 学习率策略 ----------------
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0, power=0.9, begin=0, end=160000, by_epoch=False)
]

# -------------------- 训练循环 -----------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# -------------------- Hooks -----------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2000, by_epoch=False),
    logger=dict(type='LoggerHook', interval=50)
)

log_level = 'INFO'
