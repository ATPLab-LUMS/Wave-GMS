# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Purpose                   : Downsample2D + DiagonalGaussianDistribution utilities
# Usage                     : from networks.novel.lite_vae.utils import Downsample2D, DiagonalGaussianDistribution
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__                         import annotations

import torch
import torch.nn.functional              as F
from torch                              import nn, Tensor
# ------------------------------------------------------------------------------#


# -------------------------------- Downsample2D --------------------------------#
class Downsample2D(nn.Module):
    """2× (or arbitrary) spatial downsampler.

    If `learnable=True`, uses a strided conv (kernel=3, stride=scale_factor).
    Otherwise uses average pooling. Channels are preserved.
    """
    def __init__(self, channels: int, scale_factor: int = 2, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=scale_factor, padding=1, bias=False)
        else:
            self.op = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


# ------------------- Diagonal Gaussian Distribution (VAE) ---------------------#
class DiagonalGaussianDistribution:
    """Reparameterisation helper and KL divergence for N(mu,σ^2) vs N(0,1).

    params: concatenated [mu, logvar] with shape (B, 2*C, H, W).
    """
    def __init__(
        self,
        params       : Tensor,
        device       : torch.device | str | None = None,
        dtype        : torch.dtype                = torch.float32,
        deterministic: bool                       = False,
    ) -> None:
        device               = device or ("cuda" if torch.cuda.is_available() else "cpu")
        mu, logvar           = torch.chunk(params, 2, dim=1)

        self.mu              = mu.to(device=device, dtype=dtype)
        self.logvar          = torch.clamp(logvar, min=-30.0, max=20.0).to(device=device, dtype=dtype)
        self.var             = torch.exp(self.logvar)
        self.std             = torch.exp(0.5 * self.logvar)

        self.device          = device
        self.dtype           = dtype
        self.deterministic   = deterministic

        if self.deterministic:
            zeros           = torch.zeros_like(self.mu, device=device, dtype=dtype)
            self.var        = zeros
            self.std        = zeros

    def sample(self) -> Tensor:
        """Reparameterised sample: mu + std * eps."""
        eps = torch.randn_like(self.std, device=self.device, dtype=self.std.dtype)
        return self.mu + eps * self.std

    def mode(self) -> Tensor:
        """Deterministic mode (mean)."""
        return self.mu

    def kl(self) -> Tensor:
        """Pixel-wise KL divergence; caller can reduce as needed."""
        if self.deterministic:
            return torch.zeros(1, device=self.device, dtype=self.dtype)
        return 0.5 * torch.sum(self.mu.pow(2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
# ------------------------------------------------------------------------------#