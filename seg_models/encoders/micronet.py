from ..micronet_backbone import MicroNet
from ..micronet_backbone import cfg
import torch.nn as nn
from copy import deepcopy
from ._base import EncoderMixin


class MicroNetEncoder(MicroNet, EncoderMixin):
    def __init__(self, out_channels, mode=0, depth=5, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode # m0, m1, m2, m3 have different stages
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

    def get_stages(self):
        if self.mode == 0:
            stages = [
                nn.Identity(),
                self.features[:1],
                self.features[1:2],
                self.features[2:3],
                self.features[3:5],
                self.features[5:],
            ]
        elif self.mode == 1:
            stages = [
                nn.Identity(),
                self.features[:1],
                self.features[1:2],
                self.features[2:4],
                self.features[4:7],
                self.features[7:],
            ]
        else:
            stages = [
                nn.Identity(),
                self.features[:1],
                self.features[1:2],
                self.features[2:4],
                self.features[4:8],
                self.features[8:],
            ]
        return stages

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

# micronet code configuration from authoers' code
cfg_m0 = deepcopy(cfg)
cfg_m0_list = ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock',
               'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy6_exp4_4M_221',
               'MODEL.MICRONETS.STEM_CH', '4',
               'MODEL.MICRONETS.STEM_GROUPS', '2,2',
               'MODEL.MICRONETS.STEM_DILATION', '1',
               'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf',
               'MODEL.MICRONETS.OUT_CH', '640',
               'MODEL.MICRONETS.DEPTHSEP', 'True',
               'MODEL.MICRONETS.POINTWISE', 'group',
               'MODEL.MICRONETS.DROPOUT', '0.05',
               'MODEL.ACTIVATION.MODULE', 'DYShiftMax',
               'MODEL.ACTIVATION.ACT_MAX', '2.0',
               'MODEL.ACTIVATION.LINEARSE_BIAS', 'False',
               'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0',
               'MODEL.ACTIVATION.INIT_A', '1.0,1.0',
               'MODEL.ACTIVATION.INIT_B', '0.0,0.0',
               'MODEL.ACTIVATION.REDUCTION', '8',
               'MODEL.MICRONETS.SHUFFLE', 'True']
cfg_m0.merge_from_list(cfg_m0_list)

cfg_m1 = deepcopy(cfg)
cfg_m1_list = ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock',
               'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy6_exp6_6M_221',
               'MODEL.MICRONETS.STEM_CH', '6',
               'MODEL.MICRONETS.STEM_GROUPS', '3,2',
               'MODEL.MICRONETS.STEM_DILATION', '1',
               'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf',
               'MODEL.MICRONETS.OUT_CH', '960',
               'MODEL.MICRONETS.DEPTHSEP', 'True',
               'MODEL.MICRONETS.POINTWISE', 'group',
               'MODEL.MICRONETS.DROPOUT', '0.05',
               'MODEL.ACTIVATION.MODULE', 'DYShiftMax',
               'MODEL.ACTIVATION.ACT_MAX', '2.0',
               'MODEL.ACTIVATION.LINEARSE_BIAS', 'False',
               'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0',
               'MODEL.ACTIVATION.INIT_A', '1.0,1.0',
               'MODEL.ACTIVATION.INIT_B', '0.0,0.0',
               'MODEL.ACTIVATION.REDUCTION', '8',
               'MODEL.MICRONETS.SHUFFLE', 'True']
cfg_m1.merge_from_list(cfg_m1_list)

cfg_m2 = deepcopy(cfg)
cfg_m2_list = ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock',
               'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy9_exp6_12M_221',
               'MODEL.MICRONETS.STEM_CH', '8',
               'MODEL.MICRONETS.STEM_GROUPS', '4,2',
               'MODEL.MICRONETS.STEM_DILATION', '1',
               'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf',
               'MODEL.MICRONETS.OUT_CH', '1024',
               'MODEL.MICRONETS.DEPTHSEP', 'True',
               'MODEL.MICRONETS.POINTWISE', 'group',
               'MODEL.MICRONETS.DROPOUT', '0.1',
               'MODEL.ACTIVATION.MODULE', 'DYShiftMax',
               'MODEL.ACTIVATION.ACT_MAX', '2.0',
               'MODEL.ACTIVATION.LINEARSE_BIAS', 'False',
               'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0',
               'MODEL.ACTIVATION.INIT_A', '1.0,1.0',
               'MODEL.ACTIVATION.INIT_B', '0.0,0.0',
               'MODEL.ACTIVATION.REDUCTION', '8',
               'MODEL.MICRONETS.SHUFFLE', 'True']

cfg_m2.merge_from_list(cfg_m2_list)

cfg_m3 = deepcopy(cfg)
cfg_m3_list = ['MODEL.MICRONETS.BLOCK', 'DYMicroBlock',
               'MODEL.MICRONETS.NET_CONFIG', 'msnx_dy12_exp6_20M_020',
               'MODEL.MICRONETS.STEM_CH', '12',
               'MODEL.MICRONETS.STEM_GROUPS', '4,3',
               'MODEL.MICRONETS.STEM_DILATION', '1',
               'MODEL.MICRONETS.STEM_MODE', 'spatialsepsf',
               'MODEL.MICRONETS.OUT_CH', '1024',
               'MODEL.MICRONETS.DEPTHSEP', 'True',
               'MODEL.MICRONETS.POINTWISE', 'group',
               'MODEL.MICRONETS.DROPOUT', '0.1',
               'MODEL.ACTIVATION.MODULE', 'DYShiftMax',
               'MODEL.ACTIVATION.ACT_MAX', '2.0',
               'MODEL.ACTIVATION.LINEARSE_BIAS', 'False',
               'MODEL.ACTIVATION.INIT_A_BLOCK3', '1.0,0.0',
               'MODEL.ACTIVATION.INIT_A', '1.0,0.5',
               'MODEL.ACTIVATION.INIT_B', '0.0,0.5',
               'MODEL.ACTIVATION.REDUCTION', '8',
               'MODEL.MICRONETS.SHUFFLE', 'True']
cfg_m3.merge_from_list(cfg_m3_list)

micronet_encoders = {
    "micronet_m0": {
        "encoder": MicroNetEncoder,
        "params": {
            "out_channels": (3, 4, 8, 12, 32, 384),
            "mode": 0,
            "cfg": cfg_m0
        }
    },
    "micronet_m1": {
        "encoder": MicroNetEncoder,
        "params": {
            "out_channels": (3, 6, 8, 16, 32, 576),
            "mode": 0,
            "cfg": cfg_m1
        }
    },
    "micronet_m2": {
        "encoder": MicroNetEncoder,
        "params": {
            "out_channels": (3, 8, 12, 24, 64, 768),
            "mode": 1,
            "cfg": cfg_m2
        }
    },
    "micronet_m3": {
        "encoder": MicroNetEncoder,
        "params": {
            "out_channels": (3, 12, 16, 24, 80, 864),
            "mode": 2,
            "cfg": cfg_m3
        }
    },
}
