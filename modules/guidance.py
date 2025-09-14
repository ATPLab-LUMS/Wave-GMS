# ------------------------------------------------------------------------------#
#
# File name                 : guidance.py
# Purpose                   : Guidance feature providers (edge, wavelet, DINO) + SKFF fusion
# Usage                     : from networks.guidance import prepare_guidance, SKFF
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 25, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import io, math, logging, contextlib

from typing import List

import cv2
import numpy                as np
import torch
import torch.nn             as nn
import torch.nn.functional  as F

# If you keep HaarTransform under novel.lite_vae.blocks
from modules.novel.lite_vae.blocks.haar import HaarTransform  # adjust path if needed
                                             #
# ------------------------------------------------------------------------------#
#                         Wavelet Sub-bands                                      #
# ------------------------------------------------------------------------------#
def get_wavelet_subbands(images: torch.Tensor, lv: int = 1, drop_approx: bool = True) -> torch.Tensor:
    """
    Returns wavelet sub-bands. If drop_approx=True, removes the approximation channel(s).
    """
    wavelet_fn  = HaarTransform()
    sub_bands   = wavelet_fn.dwt(images, level=lv) / 2.0
    if drop_approx:
        sub_bands = sub_bands[:, 3:, :, :]   # drop approximation (first 3)
    return sub_bands

#
#                         Guidance Router                                        #
# ------------------------------------------------------------------------------#
def prepare_guidance(image: torch.Tensor, mode: str = "edge") -> torch.Tensor:
    """
    mode: 'edge' | 'wavelet' | 'dino'
    Returns a guidance tensor on CPU/GPU matching the input device.
    """
    if mode == "wavelet":
        return get_wavelet_subbands(image)

    raise ValueError(f"Unknown guidance mode: {mode}")

# ------------------------------------------------------------------------------#
#                         Selective Kernel Feature Fusion (SKFF)                #
# ------------------------------------------------------------------------------#

class SKFF(nn.Module):
    """Selective Kernel Feature Fusion.

    Collapses multiple streams of equal channel width into one fused output.

    Input:  [B, streams*C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(
        self,
        *,
        channels: int,
        reduction: int = 8,
        streams: int = 3,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if streams < 2:
            raise ValueError(f"streams must be >= 2, got {streams}.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")

        self.channels = int(channels)
        self.streams = int(streams)
        reduced = max(1, self.channels // int(reduction))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Sequential(
            nn.Conv2d(self.channels, reduced, kernel_size=1, bias=bias),
            nn.PReLU(),
        )
        self.expands = nn.ModuleList(
            [nn.Conv2d(reduced, self.channels, kernel_size=1, bias=bias) for _ in range(self.streams)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, total_c, h, w = x.shape
        expected_total = self.streams * self.channels
        
        if total_c != expected_total:
            raise ValueError(
                f"Expected {expected_total} channels but got {total_c}."
            )

        # Split into streams [B, C, H, W] each
        chunks: List[torch.Tensor] = list(torch.chunk(x, self.streams, dim=1))

        fused = torch.stack(chunks, dim=0).sum(dim=0)  # [B, C, H, W]
        z = self.reduce(self.gap(fused))               # [B, reduced, 1, 1]

        scores = torch.stack([head(z) for head in self.expands], dim=1)  # [B, S, C, 1, 1]
        weights = F.softmax(scores, dim=1)                               # [B, S, C, 1, 1]

        out = sum(weights[:, i] * chunks[i] for i in range(self.streams))  # [B, C, H, W]
        return out


class SKFFStreamwise(nn.Module):

    def __init__(
        self,
        *,
        channels: int,
        reduction: int = 8,
        streams: int = 3,
        bias: bool = False,
        spatial: bool = True,
        return_fused: bool = False,
    ) -> None:
        super().__init__()
        if streams < 2:
            raise ValueError(f"streams must be >= 2, got {streams}.")
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")

        self.channels = int(channels)
        self.streams = int(streams)
        self.spatial = bool(spatial)
        self.return_fused = bool(return_fused)
        reduced = max(1, self.channels // int(reduction))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Sequential(
            nn.Conv2d(self.channels, reduced, kernel_size=1, bias=bias),
            nn.PReLU(),
        )
        self.expands = nn.ModuleList(
            [nn.Conv2d(reduced, 1, kernel_size=1, bias=bias) for _ in range(self.streams)]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got shape {tuple(x.shape)}")
        b, total_c, h, w = x.shape
        expected_total = self.streams * self.channels
        if total_c != expected_total:
            raise ValueError(
                "Input channel mismatch: got "
                f"{total_c} but expected streams*channels={self.streams}*{self.channels}={expected_total}."
            )

        # Split into streams and aggregate context
        chunks = torch.chunk(x, self.streams, dim=1)  # tuple of [B, C, H, W]
        fused = torch.stack(chunks, dim=0).sum(dim=0)  # [B, C, H, W]
        z = self.reduce(self.gap(fused))               # [B, reduced, 1, 1]

        # Per-stream logits -> softmax over streams
        logits = torch.stack([head(z) for head in self.expands], dim=1)  # [B, S, 1, 1, 1]
        logits = logits.squeeze(2)  # [B, S, 1, 1]
        weights = F.softmax(logits, dim=1)  # [B, S, 1, 1]

        if self.spatial:
            weights = weights.expand(b, self.streams, h, w)  # broadcast to spatial

        if not self.return_fused:
            return weights  # [B, S, H, W] or [B, S, 1, 1]

        # Build per-stream fused maps: collapse C->1 via mean, then scale by weight
        fused_maps = []
        for i in range(self.streams):
            # mean over channel dimension for the i-th stream -> [B, H, W]
            base = chunks[i].mean(dim=1)  # [B, H, W]
            wi = weights[:, i]
            if wi.dim() == 3:
                fused_maps.append(base * wi)  # [B, H, W]
            else:
                # wi is [B,1,1] -> broadcast
                fused_maps.append(base * wi.expand_as(base))
        fused_per_stream = torch.stack(fused_maps, dim=1)  # [B, S, H, W]

        return weights, fused_per_stream

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"channels={self.channels}, streams={self.streams}, spatial={self.spatial}, "
            f"return_fused={self.return_fused}"
        )


if __name__ == "__main__":  # Basic smoke tests
    # Original SKFF collapses streams to channels
    _x = torch.randn(2, 3 * 48, 32, 32)
    _m = SKFF(channels=48, reduction=8, streams=3)
    _y = _m(_x)
    print("SKFF:", tuple(_y.shape))

    # Stream-wise variant returns one map per stream
    _x2 = torch.randn(2, 4 * 3, 16, 16)  # 4 RGB streams
    _g = SKFFStreamwise(channels=3, streams=4, spatial=True, return_fused=True)
    _w = _g(_x2)
    if isinstance(_w, tuple):
        _weights, _fused = _w
        print("SKFFStreamwise weights:", tuple(_weights.shape))
        print("SKFFStreamwise fused:  ", tuple(_fused.shape))
    else:
        print("SKFFStreamwise:", tuple(_w.shape))
