# ------------------------------------------------------------------------------#
#
# File name                 : utils.py
# Purpose                   : LiteVAE utilities (downsampling and diagonal Gaussian)
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
import torch.nn.functional     as F

from typing                    import Optional
from torch                     import nn, Tensor
# ------------------------------------------------------------------------------#


# ------------------------------- Downsample2D ---------------------------------#
class Downsample2D(nn.Module):
    """
    2D spatial downsampler by an integer `scale_factor`.

    If `learnable=True`, uses a strided Conv2d (k=3, s=scale_factor).
    Else, uses AvgPool2d with kernel=stride=scale_factor.
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
                in_channels  = channels,
                out_channels = channels,
                kernel_size  = 3,
                stride       = scale_factor,
                padding      = 1,
                bias         = False,
            )
        else:
            self.op = nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        return self.op(x)


# ---------------------- Diagonal Gaussian Distribution ------------------------#
class DiagonalGaussianDistribution:
    """
    Helper for VAE reparameterization and KL divergence. (Adapted from [https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py#L691])
    """

    def __init__(
        self,
        params: Tensor,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        deterministic: bool = False,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        mu, logvar = torch.chunk(params, 2, dim=1)
        self.mu        = mu.to(device=device, dtype=dtype)
        self.logvar    = torch.clamp(logvar, min=-30.0, max=20.0).to(device=device, dtype=dtype)
        self.var       = torch.exp(self.logvar)
        self.std       = torch.exp(0.5 * self.logvar)

        self.device        = torch.device(device)
        self.dtype         = dtype
        self.deterministic = deterministic

        if self.deterministic:
            # Zero variance/std for deterministic mode
            z = torch.zeros_like(self.mu, device=self.device, dtype=self.dtype)
            self.var = z
            self.std = z

    # --------------------------------------------------------------------------#
    def sample(self) -> Tensor:
        """Reparameterized sample: mu + eps * std."""
        if self.deterministic:
            return self.mu
        eps = torch.randn_like(self.std, device=self.device, dtype=self.dtype)
        return self.mu + eps * self.std

    def mode(self) -> Tensor:
        """Deterministic mode (mean)."""
        return self.mu

    def kl(self) -> Tensor:
        """
        KL divergence KL( N(mu, sigma) || N(0, I) ) reduced over C,H,W.
        Returns: (B,) tensor (sum over dims, no mean).
        """
        if self.deterministic:
            return torch.zeros(self.mu.size(0), device=self.device, dtype=self.dtype)
        return 0.5 * torch.sum(
            self.mu.pow(2) + self.var - 1.0 - self.logvar,
            dim=(1, 2, 3),
        )

# ------------------------------------------------------------------------------#