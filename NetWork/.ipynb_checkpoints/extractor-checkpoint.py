import torch.nn as nn

from NetWork.operator import l2_norm

"""
提取Group Feature
operation:
Conv(16,5,1)-InstanceNorm-ReLu-
Conv(32,5,1)-InstanceNorm-ReLu-AvgPool(2,2)
Conv(32,5,1)-InstanceNorm-ReLu-
Conv(32,5,1)-InstanceNorm-L2Norm-
"""


class VanillaCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 5, 1, 2, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 5, 1, 2, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 5, 1, 2, bias=False),
            nn.InstanceNorm2d(32)
        )

        def forward(self, x):
            x = self.conv1(self.conv0(x))
            x = l2_norm(x, axis=1)
            return x
