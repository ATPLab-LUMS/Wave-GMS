# ------------------------------------------------------------------------------#
#
# File name                 : resblock.py
# Purpose                   : Residual block (GroupNorm + Conv) utilities
# Usage                     : from networks.novel.lite_vae.blocks.resblock import ResBlock
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__ import annotations

from typing                     import Optional
import torch
import torch.nn                 as nn
from torch                      import Tensor
# ------------------------------------------------------------------------------#


# --------------------------------- Utilities ----------------------------------#
def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "silu": return nn.SiLU()
    if name == "mish": return nn.Mish(inplace=True)
    if name == "gelu": return nn.GELU()
    raise ValueError(f"Unknown activation {name}")


# ---------------------------------- ResBlock ----------------------------------#
class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU and 3×3 padded convs.

    Args:
        in_channels     : input channels
        dropout         : dropout prob
        out_channels    : output channels (default: = in_channels)
        use_conv        : if True and channel mismatch, use 3×3 skip; else 1×1
        activation      : 'silu' | 'relu' | 'mish' | 'gelu'
        norm_num_groups : GroupNorm groups (channels should be divisible)
        scale_factor    : residual scaling factor (default 1.0)
    """
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        activation: str = "silu",
        norm_num_groups: int = 8,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        out_channels     = out_channels or in_channels
        self.scale       = scale_factor

        self.norm_in     = nn.GroupNorm(norm_num_groups, in_channels)
        self.act_in      = get_activation(activation)
        self.conv_in     = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm_out    = nn.GroupNorm(norm_num_groups, out_channels)
        self.act_out     = get_activation(activation)
        self.dropout     = nn.Dropout(dropout)
        self.conv_out    = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip        = nn.Identity() if out_channels == in_channels else (
                           nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) if use_conv
                           else nn.Conv2d(in_channels, out_channels, kernel_size=1)
                           )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(self.act_in(self.norm_in(x)))
        h = self.conv_out(self.dropout(self.act_out(self.norm_out(h))))
        return (self.skip(x) + h) / self.scale
# ------------------------------------------------------------------------------#