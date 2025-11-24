

_base_ = [
    './_base_/datasets/my_dataset.py',   # 保留你自己的数据集
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

# ---------------- 模型 ----------------
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,

    # ---------------- Transformer Backbone ----------------
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

    # ---------------- FPN Neck ----------------
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=128,  # 降低通道减少显存占用
        num_outs=4,
        start_level=0,
        add_extra_convs='on_output'
    ),

    # ---------------- UFPN / UNet-style Decode Head ----------------
    decode_head=dict(
        type='FPNHead',
        in_channels=[128, 128, 128, 128],
        in_index=[0, 1, 2, 3],
        channels=128,
        num_classes=2,
        feature_strides=[4, 8, 16, 32],  # 这里是 FPN 每层对应的下采样倍数
        dropout_ratio=0.1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),


    )
    ,

# ---------------- 精简辅助 Head ----------------
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=1,  # FPN 中间层监督
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),

    # ---------------- Training & Testing ----------------
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # 可改为 'slide' + crop_size/stride
)

# ---------------- 优化器 ----------------
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001
    ),
    clip_grad=None
)
# ---------------- Evaluation Metrics ----------------
val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'],      # 计算 IoU & mIoU
)

# 添加 F1、Precision、Recall
test_evaluator = dict(
    type='FscoreMetric',
    beta=1.0,                 # F1 score
    average='none',           # 分类别显示
    score_mode='binary',      # 二分类（Crack vs background）
)
