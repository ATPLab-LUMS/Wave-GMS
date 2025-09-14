# ------------------------------------------------------------------------------#
#
# File name                 : midblock.py
# Purpose                   : MidBlock2D — two residual layers (UNet middle)
# Usage                     : from networks.novel.lite_vae.blocks.midblock import MidBlock2D
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

import torch
import torch.nn                   as nn
from torch                        import Tensor

from modules.novel.lite_vae.blocks.resblock import ResBlock
# ------------------------------------------------------------------------------#


# --------------------------------- MidBlock2D ---------------------------------#
class MidBlock2D(nn.Module):
    """UNet mid block with 2 residual layers."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.res0 = ResBlock(in_channels=in_channels,  out_channels=out_channels, dropout=dropout)
        self.res1 = ResBlock(in_channels=out_channels, out_channels=out_channels, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res0(x)
        x = self.res1(x)
        return x
# ------------------------------------------------------------------------------#