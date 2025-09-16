# ------------------------------------------------------------------------------#
#
# File name                 : midblock.py
# Purpose                   : Defines UNet mid-block with residual layers, 
#                             optionally using SMC-enabled ResBlocks.
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
import torch

from torch import nn, Tensor

from modules.novel.lite_vae.blocks.resblock import ResBlock
# ------------------------------------------------------------------------------#


# ------------------------------- MidBlock -------------------------------------#
class MidBlock2D(nn.Module):
    """
    UNet mid-block with two sequential residual layers.
    Can optionally use SMC-enabled ResBlocks.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.res0      = ResBlock(in_channels=in_channels, out_channels=out_channels, dropout=dropout)
        self.res1      = ResBlock(in_channels=out_channels, out_channels=out_channels, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res0(x)
        x = self.res1(x)
        return x