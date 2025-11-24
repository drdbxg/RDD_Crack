import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from .module import CBAM, DeformConvModule  # 你放在 module.py

@MODELS.register_module()
class MyUNet(BaseModule):
    def __init__(self, out_channels=64, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        c = 64

        # Encoder
        self.enc1 = self._make_layer(3, c)
        self.enc2 = self._make_layer(c, c*2, stride=2)
        self.enc3 = self._make_layer(c*2, c*4, stride=2)
        self.enc4 = self._make_layer(c*4, c*8, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DeformConvModule(c*8, c*16, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU')),
            self._make_layer(c*16, c*16),
            CBAM(c*16)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(c*16, c*8, 2, stride=2)
        self.dec4 = self._make_layer(c*16, c*8)

        self.up3 = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3 = self._make_layer(c*8, c*4)

        self.up2 = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = self._make_layer(c*4, c*2)

        self.up1 = nn.ConvTranspose2d(c*2, c, 2, stride=2)
        self.dec1 = self._make_layer(c*2, c)

        # 输出给 decode_head
        self.out = nn.Conv2d(c, out_channels, 1)

    def _make_layer(self, in_c, out_c, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            CBAM(out_c)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d = self.up4(b)
        e4 = F.interpolate(e4, size=d.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, e4], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        e3 = F.interpolate(e3, size=d.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, e3], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        e2 = F.interpolate(e2, size=d.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, e2], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        e1 = F.interpolate(e1, size=d.shape[2:], mode='bilinear', align_corners=False)
        d = torch.cat([d, e1], dim=1)
        d = self.dec1(d)

        out = self.out(d)  # 输出给 decode_head
        return out
