# ------------------------------------------------------------------------------#
#
# File name                 : litevae.py
# Purpose                   : LiteVAE wrapper (encoder → moments → DiagonalGaussianDistribution)
# Usage                     : from networks.novel.lite_vae.litevae import LiteVAE
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__                             import annotations

from typing                                  import Literal
import torch
import torch.nn                               as nn

from modules.novel.lite_vae.encoder          import LiteVAEEncoder
from modules.novel.lite_vae.utils            import DiagonalGaussianDistribution
from modules.novel.lite_vae.blocks.lip       import LIPBlock
# ------------------------------------------------------------------------------#


# ------------------------------------ LiteVAE ---------------------------------#
class LiteVAE(nn.Module):
    """
    Encoder-only default. If decode-path is enabled by the caller, it should supply
    a compatible decoder externally (e.g., TinyVAE/SD-VAE).
    """
    def __init__(
        self,
        encoder     : LiteVAEEncoder = LiteVAEEncoder(),
        latent_dim  : int             = 4,
        output_type : Literal["image", "wavelet"] = "image",  # kept for future decode hook
        use_1x1_conv: bool            = False,
        decode      : bool            = False,
    ) -> None:
        super().__init__()
        assert output_type in ["image", "wavelet"]
        self.encoder      = encoder
        self.output_type  = output_type
        self.decode_flag  = decode

        pre_ch            = latent_dim * 2    # [mu|logvar]
        post_ch           = latent_dim        # z
        self.pre_conv     = nn.Conv2d(pre_ch,  pre_ch,  1) if use_1x1_conv else nn.Identity()
        self.post_conv    = nn.Conv2d(post_ch, post_ch, 1) if use_1x1_conv else nn.Identity()

        # optional LIP downsample (kept off by default in your pipeline)
        self.lip          = LIPBlock(in_channels=latent_dim, p=2)

    # --------------------------------------------------------------------------#
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.pre_conv(self.encoder(image))

    # --------------------------------------------------------------------------#
    def forward(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor | dict:
        moments  = self.encode(image).to(device=image.device, dtype=image.dtype)
        posterior= DiagonalGaussianDistribution(moments, device=image.device, dtype=image.dtype)
        z        = posterior.sample() if sample else posterior.mode()
        # z      = self.lip(z)  # enable if you want 2× downsampled latents

        if self.decode_flag:
            # Decoder integration is external; left intentionally unimplemented here.
            return {
                "latent"      : z,
                "latent_dist" : posterior,
                "kl_reg"      : posterior.kl().mean(),  # scalar
            }
        return z
# ------------------------------------------------------------------------------#