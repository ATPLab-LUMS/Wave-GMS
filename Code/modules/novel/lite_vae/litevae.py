# ------------------------------------------------------------------------------#
#
# File name                 : litevae.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/litevae.py
# -----------------------------------------------------------------------------
# LiteVAE full model wrapper
#   - Encoder: multi-level DWT + UNet-based extractor and aggregator
#   - Decoder: imported from pretrained HuggingFace VAE (Tiny or KL)
#   - Output: image or wavelet + VAE KL + latents
# -----------------------------------------------------------------------------


import torch
from torch  import nn
from typing import Literal

from __future__                         import annotations
from .encoder                           import LiteVAEEncoder
from .utils                             import DiagonalGaussianDistribution
from diffusers                          import AutoencoderTiny
from networks.novel.lite_vae.blocks.lip import LIPBlock

class LiteVAE(nn.Module):
    def __init__(
        self,
        encoder: LiteVAEEncoder = LiteVAEEncoder(),
        decoder: None | AutoencoderTiny = AutoencoderTiny(),
        latent_dim: int = 4,
        output_type: Literal["image", "wavelet"] = "image",
        use_1x1_conv: bool = False,
        decode : bool = False,
    ) -> None:
        super().__init__()
        assert output_type in ["image", "wavelet"]

        self.encoder = encoder
        self.decoder = decoder
        self.wavelet_fn = encoder.wavelet_fn
        self.output_type = output_type
        self.decode = decode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 1x1 convs to match latent dimension
        pre_channels  = latent_dim * 2  # For [mu, logvar]
        post_channels = latent_dim     # Actual decoded latent channels

        self.lip_block = LIPBlock(in_channels = latent_dim, p = 2).to(device = self.device)
        
        self.pre_conv  = nn.Conv2d(pre_channels, pre_channels, 1) if use_1x1_conv else nn.Identity()
        self.post_conv = nn.Conv2d(post_channels, post_channels, 1) if use_1x1_conv else nn.Identity()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.pre_conv(self.encoder(image))

    def decode(self, latent: torch.Tensor, scale_factor = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.post_conv(latent)

        if self.output_type == "image":
            # decode img bring it to 1 channel and the clamp it before reconstructing wavelets?
            image_recon   = self.decoder((1 / scale_factor) * latent)
            image_recon   = torch.mean(image_recon, dim=1, keepdim=True)
            image_recon   = torch.clamp((image_recon + 1.0) / 2.0, min=0.0, max=1.0)
            wavelet_recon = self.wavelet_fn.dwt(image_recon[:, 0: 1, :], level = 1) / 2 # indexing image_recon from (B, C, H, W) to 
                                                                            # (B, 1, H, W) i.e. choose any channel 
                                                                            # so that the wavelet recon is of shape
                                                                            # (B, # of sub-bands, H/(2 ^ l), W/(2 ^ l)
                                                                            # which matches the shape of the sub-bands 
                                                                            # gotten from forward method
        else:
            wavelet_recon = self.decoder(latent)
            image_recon   = self.wavelet_fn.idwt(wavelet_recon, level = 1) * 2

        return image_recon, wavelet_recon

    def forward(self, image: torch.Tensor, sample: bool = True) -> dict:
        # Ideally the loss will be computed as something like L_train = L1(image_recon, image) + 
        #                                                               Charboneir Loss(wavelet_recon, wavelet) +
        #                                                               lambda_reg * kl_reg
        # but we are feeding it as an encoder only model in the GMS pipelines
        latents_raw = self.encode(image).to(device=image.device, dtype = image.dtype)
        latent_dist = DiagonalGaussianDistribution(latents_raw)
        latent = latent_dist.sample() if sample else latent_dist.mode()
        kl_reg = latent_dist.kl().mean()

        # Use LIP to downsample latent from (B, C, H, W) to (B, C, H/p, W/p) where p = 2
        # latent    = self.lip_block(latent)
        
        if self.decode and self.decoder is not None:
            image_recon, wavelet_recon = self.decode(latent)
            return {
                "sample": image_recon,
                "wavelet": wavelet_recon,
                "latent": latent,
                "kl_reg": kl_reg,
                "latent_dist": latent_dist,
            }
        else:
            return latent
           
                
