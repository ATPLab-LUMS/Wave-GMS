# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/utils.py
# -----------------------------------------------------------------------------
# Utility layers and helpers shared across LiteVAE modules.
#   • Downsample2D : non‑learnable (nearest‑avg) or learnable strided‑conv downsampler
#   • DiagonalGaussianDistribution : VAE reparameterisation helper + KL
# -----------------------------------------------------------------------------

import torch
from __future__ import annotations
from torch      import nn, Tensor

__all__ = [
    "Downsample2D",
    "DiagonalGaussianDistribution",
]

# -----------------------------------------------------------------------------
# Downsample2D -----------------------------------------------------------------
# -----------------------------------------------------------------------------

class Downsample2D(nn.Module):
    """2× (or arbitrary) spatial downsampler.

    If `learnable=True`, uses a strided conv (kernel=3, stride=scale_factor).
    Otherwise uses average pooling + optional 1×1 conv for channel projection.
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        self.scale = scale_factor

        if learnable:
            self.op = nn.Conv2d(
                channels,
                channels,
                kernel_size = 3,
                stride      = scale_factor,
                padding     = 1,
                bias        = False,
            )
        else:
            self.op = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


# -----------------------------------------------------------------------------
# Diagonal Gaussian Distribution ----------------------------------------------
# -----------------------------------------------------------------------------

class DiagonalGaussianDistribution:
    """Helper class for VAE reparameterisation and KL divergence.

    Parameters
    ----------
    params : Tensor
        Concatenated tensor `[mu, logvar]` with shape `(B, 2*C, H, W)`.
    """

    def __init__(self, params: Tensor, device = 'cuda' if torch.cuda.is_available() else 'cpu', dtype = torch.float32, deterministic = False) -> None:
        
        self.mu, self.logvar = torch.chunk(params, 2, dim=1)
        self.mu     = self.mu.to(device = device, dtype = dtype)
        self.logvar = torch.clamp(self.logvar, min=-30.0, max=20.0).to(device = device, dtype = dtype)  # Prevent extreme values
        self.var    = torch.exp(self.logvar).to(device = device, dtype = dtype) 
        self.std    = torch.exp(0.5 * self.logvar).to(device = device, dtype = dtype) 
        
        self.device = device
        self.dtype = dtype
        self.deterministic = deterministic

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mu, device = self.device, dtype = self.dtype)
        
    def sample(self) -> Tensor:
        """Reparameterised sample."""
        eps = torch.randn_like(self.std).to(dtype = torch.float32, device = self.device)
        return self.mu + eps * self.std

    def mode(self) -> Tensor:
        """Deterministic mode (mean)."""
        return self.mu

    def kl(self) -> Tensor:
        """Pixel‑wise KL divergence N(mu,σ) || N(0,1).
        Returns tensor of same spatial shape (averaging up to caller).
        """
        if self.deterministic:
            return torch.Tensor([0.0])
        return 0.5 * torch.sum(torch.pow(self.mu, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        diff = prediction - target
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon**2))
        return loss
    """ 
    Example Usage: Level 1 sub-bands compared
    criterion = CharbonnierLoss(epsilon=1e-3)
    loss = criterion(wavelet_recon, haar_transform_out_list[0])
    """
   