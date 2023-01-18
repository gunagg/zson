import numpy as np
import torch
import torch.nn as nn
from habitat_baselines.rl.ddppo.policy.resnet import BasicBlock, Bottleneck
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import RunningMeanAndVar

from zson.models import resnet_gn, resnet_zer


def _get_backbone(name: str, in_channels: int, baseplanes: int) -> nn.Module:
    if name in ["resnet18", "resnet50"]:
        return resnet_gn.__dict__[name](in_channels, baseplanes, baseplanes // 2)
    elif name in ["resnet9"]:
        return resnet_zer.__dict__[name](in_channels, baseplanes, baseplanes // 2)
    else:
        raise ValueError("invalid backbone: {}".format(name))


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone: str,
        in_channels: int = 3,
        baseplanes: int = 32,
        spatial_size: int = 128,
        normalize_input: bool = True,
        zero_init_residual: bool = True,
    ):
        super().__init__()
        if normalize_input:
            self.normalize = RunningMeanAndVar(in_channels)
        else:
            self.normalize = nn.Identity()

        self.backbone = _get_backbone(backbone, in_channels, baseplanes)

        final_spatial = int(spatial_size * self.backbone.final_spatial_compress)
        after_compression_flat_size = 512 if backbone == "resnet9" else 2048
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial**2))
        )
        self.compression = nn.Sequential(
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, num_compression_channels),
            nn.ReLU(True),
        )

        self.output_shape = (
            num_compression_channels,
            final_spatial,
            final_spatial,
        )
        self.output_size = np.prod(self.output_shape)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, (Bottleneck, BasicBlock)):
                    nn.init.constant_(m.convs[-1].weight, 0)

    def forward(self, x):
        x = self.normalize(x)
        x = self.backbone(x)
        x = self.compression(x)
        x = torch.flatten(x, 1)
        return x
