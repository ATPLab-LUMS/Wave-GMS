# ------------------------------------------------------------------------------#
#
# File name                 : smc.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/blocks/smc.py
# -----------------------------------------------------------------------------
# Self-Modulated Convolution (SMC) — a light replacement for GroupNorm + Conv.
# Inspired by StyleGAN2/3 weight‐modulated convolutions but simplified:
#   * Each input channel has a learned scale s_c (γ in paper)
#   * A single learned global gain g (β in paper) rescales the *output* map.
#   * No demod / no noise injection.
# -----------------------------------------------------------------------------
# Example
# -------
# >>> import torch
# >>> from networks.novel.litevae.blocks.smc import SMC
# >>> x = torch.randn(2, 64, 28, 28)
# >>> smc = SMC(64, 64, 3)
# >>> y = smc(x)
# >>> y.shape
# torch.Size([2, 64, 28, 28])
# -----------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from __future__ import annotations
from typing     import Optional
from torch      import nn, Tensor

__all__ = [
    "SMC",
]

# -----------------------------------------------------------------------------
# Helper: modulated 2-D convolution -------------------------------------------
# -----------------------------------------------------------------------------

def modulated_conv2d(
    x: Tensor,
    w: Tensor,
    s: Tensor,
    padding: int = 1,
    stride: int = 1,
    input_gain: Optional[Tensor] = None,
) -> Tensor:
    """Weight-modulated conv2d (no demod).

    Parameters
    ----------
    x : Tensor
        Input tensor (B, C_in, H, W)
    w : Tensor
        Convolution weight (C_out, C_in, kH, kW)
    s : Tensor
        Per-sample, per-in-channel scale (B, C_in)
    padding : int, default = 1
    stride : int, default = 1
    input_gain : Tensor | None
        Optional global gain γ (scalar) applied **after** convolution.
    """
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape

    # Modulate weights: w   <-   w * s_broadcast
    w = w.unsqueeze(0)                          # (1, C_out, C_in, kH, kW)
    s = s.view(B, 1, C_in, 1, 1)                # (B, 1, C_in, 1, 1)
    w_mod = w * s                               # (B, C_out, C_in, kH, kW)

    # Reshape for grouped conv (each sample is its own group)
    x = x.view(1, B * C_in, H, W)
    w_mod = w_mod.view(B * C_out, C_in, kH, kW)

    y = F.conv2d(x, w_mod, padding=padding, stride=stride, groups=B)
    y = y.view(B, C_out, y.shape[-2], y.shape[-1])

    if input_gain is not None:
        y = y * input_gain
    return y


# -----------------------------------------------------------------------------
# Main SMC layer --------------------------------------------------------------
# -----------------------------------------------------------------------------

class SMC(nn.Module):
    """Self-Modulated Convolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int | None
        Defaults to `in_channels` (depth-preserving conv).
    kernel_size : int, default = 3
    stride : int,   default = 1
    padding : int | None
        If `None`, it is set to `kernel_size // 2` (same padding).
    bias : bool, default = True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        padding = kernel_size // 2 if padding is None else padding

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

        # Learnable per-channel scale (γ) and global gain (β)
        self.scales = nn.Parameter(torch.ones(in_channels))   # (C_in,)
        self.gain   = nn.Parameter(torch.ones(1))             # scalar

        # Weight init — follow Kaiming but scaled down (like StyleGAN)
        nn.init.normal_(self.conv.weight, 0.0, 1.0)
        if bias:
            nn.init.zeros_(self.conv.bias)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        B, C_in, _, _ = x.shape
        s = self.scales.expand(B, -1)  # (B, C_in)
        out = modulated_conv2d(
            x=x,
            w=self.conv.weight,
            s=s,
            padding=self.conv.padding[0],
            stride=self.conv.stride[0],
            input_gain=self.gain,
        )
        if self.conv.bias is not None:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out