# ------------------------------------------------------------------------------#
#
# File name                 : lip.py
# Purpose                   : Local Importance Pooling (LIP) downsampler
# Usage                     : from networks.novel.lite_vae.blocks.lip import LIPBlock
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
import torch.nn                   as nn
import torch.nn.functional        as F
# ------------------------------------------------------------------------------#


# ------------------------------- LIP operator ---------------------------------#
def lip2d(x: torch.Tensor, logit: torch.Tensor, p: int = 2, margin: float = 1e-6,
          clamp_logits: float | None = None) -> torch.Tensor:
    """
    Weighted average pooling with learned importance (logits).
    Args:
        x              : (B, C, H, W)
        logit          : (B, 1, H, W)
        p              : downsample factor
        margin         : numerical stability
        clamp_logits   : optional clamp on logits before exp (e.g., 10.0). If None, no clamp.
    """
    kernel  = p
    stride  = p
    weight  = logit
    if clamp_logits is not None:
        weight = weight.clamp(min=-clamp_logits, max=clamp_logits)
    weight  = weight.exp()

    a = F.avg_pool2d(x * weight, kernel_size=kernel, stride=stride, padding=0)
    b = F.avg_pool2d(weight,       kernel_size=kernel, stride=stride, padding=0) + margin
    return a / b


# --------------------------- Bottleneck Logit Module --------------------------#
class BottleneckLogit(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------- LIP Block ------------------------------------#
class LIPBlock(nn.Module):
    def __init__(self, in_channels: int = 4, p: int = 2, clamp_logits: float | None = None):
        """
        Args:
            in_channels   : channels in x
            p             : spatial downsample factor (must divide H and W)
            clamp_logits  : optional clamp before exp for numerical stability
        """
        super().__init__()
        self.p             = p
        self.clamp_logits  = clamp_logits
        self.logit_module  = BottleneckLogit(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H % self.p == 0 and W % self.p == 0, f"H,W must be divisible by p={self.p}, got {H}x{W}"
        logits = self.logit_module(x)  # (B,1,H,W)
        return lip2d(x, logits, p=self.p, clamp_logits=self.clamp_logits)
# ------------------------------------------------------------------------------#