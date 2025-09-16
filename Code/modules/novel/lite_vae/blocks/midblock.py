# ------------------------------------------------------------------------------#
#
# File name                 : midblock.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/blocks/midblock.py
# -----------------------------------------------------------------------------
# MidBlock2D: Two residual blocks in sequence, used in UNet middle.
# -----------------------------------------------------------------------------
# Example
# -------
# >>> x = torch.randn(1, 64, 28, 28)
# >>> mid = MidBlock2D(64, 64, use_smc=False)
# >>> out = mid(x)
# >>> out.shape
# torch.Size([1, 64, 28, 28])
# -----------------------------------------------------------------------------

from __future__                              import annotations
from torch                                   import nn, Tensor
from networks.novel.lite_vae.blocks.resblock import ResBlock, ResBlockWithSMC

__all__ = [
    "MidBlock2D",
]

class MidBlock2D(nn.Module):
    """UNet Mid block with 2 residual layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        use_smc: bool = False,
    ) -> None:
        super().__init__()
        resblock_class = ResBlockWithSMC if use_smc else ResBlock
        self.res0 = resblock_class(
            in_channels  = in_channels,
            out_channels = out_channels,
            dropout      = dropout,
        )
        self.res1 = resblock_class(
            in_channels  = out_channels,
            out_channels = out_channels,
            dropout      = dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.res0(x)
        x = self.res1(x)
        return x