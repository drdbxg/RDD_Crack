import torch.nn as nn
from mmcv.cnn import build_conv_layer
from mmengine.registry import MODELS
from mmseg.models.decode_heads.fpn_head import FPNHead


@MODELS.register_module()
class FPNHeadDCN(FPNHead):
    """FPNHead using DCN in lateral and output convolutions."""

    def __init__(self, *args, **kwargs):
        super(FPNHeadDCN, self).__init__(*args, **kwargs)

    def _init_inputs(self):
        """Initialize lateral convs and fpn convs (output convs).
        Replace Conv2d with DCN-based conv layers.
        """
        super()._init_inputs()  # keep input transform settings

        # --------------- Replace lateral_convs with DCN ---------------
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            self.lateral_convs.append(
                build_conv_layer(
                    cfg=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
                    in_channels=in_channels,
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        # --------------- Replace fpn (output) convs with DCN ---------------
        self.fpn_convs = nn.ModuleList()
        for _ in self.in_channels:
            self.fpn_convs.append(
                build_conv_layer(
                    cfg=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
