#去掉 FPN，使用单层 decode head
_base_ = [
    './_base_/datasets/my_dataset.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py'
]

# ---------------- 数据预处理 ----------------
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size_divisor=32
)

# ---------------- 模型（无 FPN） ----------------
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    # Backbone（保留）
    backbone=dict(
        type='PCPVT',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/twins/pcpvt_small_20220308-e638c41c.pth'
        ),
        in_channels=3,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2
    ),

    # *** A2 设置：没有 FPN ***

    # 单层 decode head
    decode_head=dict(
        type='FCNHead',
        in_channels=512,   # backbone 第四层输出
        in_index=3,
        channels=256,
        num_convs=2,
        num_classes=2,
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        ),
    ),

    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ---------------- 优化器 ----------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
)

# ---------------- Evaluation ----------------
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(
    type='FscoreMetric', beta=1.0, average='none', score_mode='binary'
)
