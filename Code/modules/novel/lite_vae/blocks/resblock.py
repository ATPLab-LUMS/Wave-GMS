# ------------------------------------------------------------------------------#
#
# File name                 : resblock.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/lite_vae/blocks/resblock.py
# -----------------------------------------------------------------------------
# Standard Residual Block (GroupNorm + Conv) and its Self‑Modulated variant.
# -----------------------------------------------------------------------------
# Example
# -------
# >>> import torch
# >>> from networks.novel.lite_vae.blocks.resblock import ResBlock, ResBlockWithSMC
# >>> x = torch.randn(4, 32, 56, 56)
# >>> rb = ResBlock(32, out_channels=64)
# >>> y = rb(x)
# >>> y.shape  # (4, 64, 56, 56)
# torch.Size([4, 64, 56, 56])
# -----------------------------------------------------------------------------

from __future__ import annotations
from typing     import Optional
from torch      import nn, Tensor
from .smc       import SMC

__all__ = [
    "get_activation","ResBlock",
    "ResBlockWithSMC",
]

# -----------------------------------------------------------------------------
# Utilities ---------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_activation(name: str):
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


# -----------------------------------------------------------------------------
# ResBlock (GroupNorm variant) --------------------------------------------------
# -----------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with GroupNorm + SiLU activations.

    Parameters
    ----------
    in_channels : int
    dropout : float, default = 0.0
    out_channels : int | None
        If `None`, keeps `in_channels`.
    use_conv : bool, default = False
        If `True` and `in_channels != out_channels`, uses 3x3 conv for skip.
        Else uses 1x1 conv.
    activation : str, default = "silu"
    norm_num_groups : int, default = 32
    scale_factor : float, default = 1
        Optionally scales the residual sum (e.g. `sqrt(2)`).

    Note #1: Added padding = 1 to 3 x 3 convolutions to maintain spatial dimensions.
    Note #2: The in_channels and out_channels should be divisible by norm_num_groups.
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
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        self.norm_in = nn.GroupNorm(norm_num_groups, in_channels)
        self.act_in  = get_activation(activation)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)

        self.norm_out = nn.GroupNorm(norm_num_groups, out_channels)
        self.act_out  = get_activation(activation)
        self.dropout  = nn.Dropout(dropout)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

        self.scale_factor = scale_factor
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        # check shape after each layer and print it
        h = self.norm_in(x)
        h = self.act_in(h)
        h = self.conv_in(h)

        # output layers
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.dropout(h)
        h = self.conv_out(h)

        return (self.skip_connection(x) + h) / self.scale_factor

# -----------------------------------------------------------------------------
# ResBlockWithSMC --------------------------------------------------------------
# -----------------------------------------------------------------------------


class ResBlockWithSMC(nn.Module):
    """Residual block that replaces Conv layers with Self‑Modulated Conv (SMC).
    Follows the paper’s Section 4.2 design (no GroupNorm).
    """

    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.act = nn.SiLU()
        self.conv_in = SMC(in_channels, out_channels, 3)
        self.dropout = nn.Dropout(dropout)
        self.conv_out = SMC(out_channels, out_channels, 3)

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

        self.scale_factor = scale_factor

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        h = self.conv_in(self.act(x))
        h = self.conv_out(self.dropout(self.act(h)))
        return (self.skip_connection(x) + h) / self.scale_factor