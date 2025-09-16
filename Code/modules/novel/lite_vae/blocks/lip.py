# ------------------------------------------------------------------------------#
#
# File name                 : lip.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

import torch.nn.functional as F
from torch import nn

# ------------------------------------- LIP operator ------------------------------------ #
def lip2d(x, logit, p=2, margin=1e-6):
    """
    Local Importance Pooling operator.

    Args:
        x: Input tensor of shape (B, C, H, W)
        logit: Logit map of shape (B, 1, H, W)
        p: Downsampling factor (must divide H and W)
        margin: Small value added to denominator for stability

    Returns:
        Downsampled tensor of shape (B, C, H/p, W/p)
    """
    kernel = p
    stride = p
    padding = 0  # No padding needed since H and W are divisible by p

    weight = logit.exp()  # Ensure positive weights
    a = F.avg_pool2d(x * weight, kernel_size=kernel, stride=stride, padding=padding)
    b = F.avg_pool2d(weight, kernel_size=kernel, stride=stride, padding=padding) + margin

    return a / b


# --------------------------- Bottleneck Logit Module --------------------------- #
class BottleneckLogit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1)  # Output logit map
        )

    def forward(self, x):
        return self.net(x)  # Shape: (B, 1, H, W)


# ------------------------------- LIP Block Module ------------------------------ #
class LIPBlock(nn.Module):
    def __init__(self, in_channels = 4, p = 2):
        """
        Args:
            in_channels: Number of input channels
            p: Downsampling factor (must divide H and W of input)
        """
        super().__init__()
        self.p = p
        self.logit_module = BottleneckLogit(in_channels)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Downsampled tensor of shape (B, C, H/p, W/p)
        """
        B, C, H, W = x.shape
        assert H % self.p == 0 and W % self.p == 0, \
            f"Input H and W must be divisible by p={self.p}, got H={H}, W={W}"

        logits = self.logit_module(x)  # Shape: (B, 1, H, W)
        return lip2d(x, logits, p=self.p)


# ---------------------------- Example Usage ----------------------------------- #
    # # Test input
    # B, C, H, W = 12, 4, 56, 56  # H and W must be divisible by p
    # p = 2
    # x = torch.randn(B, C, H, W)

    # # Apply LIPBlock
    # lip_block = LIPBlock(in_channels=C, p = p)
    # y = lip_block(x)

    # print("Input shape:", x.shape)
    # print("Output shape:", y.shape)  # Should be (B, C, H/p, W/p)