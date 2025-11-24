import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS

# ==================== CBAM ====================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // ratio), in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


@MODELS.register_module()
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ==================== Deformable Conv (compat fallback) ====================
try:
    # preferred: mmcv.ops (fast, compiled)
    from mmcv.ops import ModulatedDeformConv2d
except Exception:
    try:
        # fallback: mmseg.ops (pure-PyTorch implementation if available)
        from mmseg.ops import ModulatedDeformConv2d
    except Exception:
        warnings.warn(
            "ModulatedDeformConv2d not found (mmcv/mmseg ops). "
            "Falling back to regular Conv2d (no deformable behavior). "
            "For best performance/install mmcv-full."
        )
        ModulatedDeformConv2d = nn.Conv2d


@MODELS.register_module()
class DeformConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()

        # 保存参数用于 forward 时切分
        self.kernel_size = kernel_size
        self.deform_groups = deform_groups

        # 输出的 offset+mask 通道数： deform_groups * 3 * k * k  (2 offsets + 1 mask)
        self.offset_channels = deform_groups * 3 * kernel_size * kernel_size

        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            self.offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )

        # 可变形卷积主体（若为普通 Conv2d，行为为普通卷积）
        self.deform_conv = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            deform_groups=deform_groups
        )

        # 归一化 + 激活（简单处理）
        self.norm = nn.BatchNorm2d(out_channels) if norm_cfg else None
        self.act = nn.ReLU(inplace=True) if act_cfg else None

        # 初始化 offset_mask_conv 为 0（让开始阶段等同于普通 conv）
        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

    def forward(self, x):
        # 生成 offset 和 mask
        offset_mask = self.offset_mask_conv(x)
        C = offset_mask.shape[1]

        # 计算切分点：前 2/3 是 offset，后 1/3 是 mask
        # 使用保留整数的方式：off_ch = offset_channels // 3 * 2
        oc = self.offset_channels
        off_ch = (oc // 3) * 2
        mask_ch = oc - off_ch

        # 安全检查：确保和实际通道数一致
        if C != oc:
            # 如果实际通道数和预期不同，尽量按比例切分（降级处理）
            off_ch = (C // 3) * 2
            mask_ch = C - off_ch

        offset = offset_mask[:, :off_ch, :, :]
        mask = offset_mask[:, off_ch:off_ch + mask_ch, :, :]
        mask = torch.sigmoid(mask)

        # 执行可变形卷积（若为普通 Conv2d 则直接调用）
        if isinstance(self.deform_conv, nn.Conv2d):
            out = self.deform_conv(x)
        else:
            # mmcv ModulatedDeformConv2d 接口通常为 (x, offset, mask)
            out = self.deform_conv(x, offset, mask)

        if self.norm is not None:
            out = self.norm(out)
        if self.act is not None:
            out = self.act(out)
        return out
