# ------------------------------------------------------------------------------#
#
# File name                 : resblock.py
# Purpose                   : Defines standard residual block with optional 
#                             dropout, normalization, and flexible skip connections.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# Note                      : Adapted from [https://arxiv.org/pdf/2405.14477.pdf]
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from typing                 import Optional

import torch

from torch                  import nn, Tensor
# ------------------------------------------------------------------------------#


# ---------------------------- Get Activation -------------------------------#
def get_activation(name: str):
    """
    Return activation function by name.
    Supported: relu, silu, mish, gelu.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "mish":
        return nn.Mish(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation {name}")
# ------------------------------------------------------------------------------#


# -------------------------------- ResBlock ------------------------------------#
class ResBlock(nn.Module):
    """
    Residual block with GroupNorm, activation, dropout, 
    and optional convolution-based skip connection.
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
        self.in_channels   = in_channels
        self.out_channels  = out_channels or in_channels

        # Input layers
        self.norm_in       = nn.GroupNorm(norm_num_groups, in_channels)
        self.act_in        = get_activation(activation)
        self.conv_in       = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

        # Output layers
        self.norm_out      = nn.GroupNorm(norm_num_groups, self.out_channels)
        self.act_out       = get_activation(activation)
        self.dropout       = nn.Dropout(dropout)
        self.conv_out      = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        # Skip connection
        if self.out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)

        self.scale_factor  = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm_in(x)
        h = self.act_in(h)
        h = self.conv_in(h)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.dropout(h)
        h = self.conv_out(h)

        return (self.skip_connection(x) + h) / self.scale_factor