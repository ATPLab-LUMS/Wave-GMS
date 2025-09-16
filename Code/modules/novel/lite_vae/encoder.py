# ------------------------------------------------------------------------------#
#
# File name                 : encoder.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/encoder.py
# -----------------------------------------------------------------------------
# LiteVAEEncoder: 3-level wavelet transform + shared feature extractors + UNet
# feature aggregation to produce VAE-style latent distribution (mu, logvar).
# -----------------------------------------------------------------------------

import torch
from torch                               import nn, Tensor
from __future__                          import annotations
from networks.novel.lite_vae.blocks      import LiteVAEUNetBlock
from networks.novel.lite_vae.blocks.haar import HaarTransform
from networks.novel.lite_vae.utils       import Downsample2D

__all__ = ["LiteVAEEncoder"]

"""
Notes:
1) The input channels to each feature extracter is 12 (page 20 of paper) and output channels is also 12. This is because from now we will use their implementation of haar transform.
The difference between their implementation and ours is that we applied the haar transform to the first channel of the RGB image across the batch resulting in (B, 4, H_l, W_l). However
their implementation uses ptwt instead of pywt which takes the whole RGB image and hence the output of each level transform is (B, 4 * 3 = 12, H_l, W_l)

2) The following is the model_channels and ch_multiples configuration for each model size of the feature extraction module
    For LiteVAE-S, model_channels = 16 and ch_multiplies = [1, 2, 2] (see unet_block.py)
    For LiteVAE-B, model_channels = 32 and ch_multiplies = [1, 2, 3]
    For LiteVAE-M, model_channels = 64 and ch_multiplies = [1, 2, 4]
    For LiteVAE-L, model_channels = 64 and ch_multiplies = [1, 2, 4]

3) Due to similar reasoning as 1), the feature aggregation module which was previously taking in_channels as in_channels * 3 and out_channels as out_channels * 2 is now going to take
   in_channels = 12 * 3 = 36 but out_channels can be flexible. The paper has used 12 (page 7 of paper). We will use 8 as the mu | logvar decomposition allows us to have the latent
   in 4 channels form which is what the LMM model expects in GMS paper

4) The following is the model_channels and ch_multiples configuration for each model size of the feature aggregation module
    For LiteVAE-S, model_channels = 16 and ch_multiplies = [1, 2, 2] (see unet_block.py)
    For LiteVAE-B, model_channels = 32 and ch_multiplies = [1, 2, 3]
    For LiteVAE-M, model_channels = 64 and ch_multiplies = [1, 2, 3]
    For LiteVAE-L, model_channels = 64 and ch_multiplies = [1, 2, 4]

5) Other Training configurations:
    i) Adam Optimizer with base LR = 10^-4 and (beta1, beta2) = (0.5, 0.9)

"""

class LiteVAEEncoder(nn.Module):
    """
    LiteVAE encoder that performs:
      - 3-level DWT to extract multi-scale high/low frequency bands
      - Shared feature extractors (UNet blocks) at each level
      - Downsampling to target resolution
      - Final aggregation UNet to produce mean+logvar

    Output shape is (B, 2C, H, W), where 2C = [mu, logvar] for Gaussian sampling.
    """

    def __init__(
        self,
        model_version: str = "litevae-s",      # Options: "litevae-s", "litevae-b", "litevae-m", "litevae-l"
        in_channels: int = 12,                # Fixed by 3-level Haar on RGB image → 3×4=12 channels
        out_channels: int = 12,               # Output feature depth after aggregation
        wavelet_fn: HaarTransform = HaarTransform(),
    ):
        super().__init__()
        self.model_version = model_version
        self.wavelet_fn    = wavelet_fn
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        # --------------------------- Model Configurations --------------------------- #
        encoder_configs = {
            "litevae-s": {"model_channels": 16, "ch_mult": [1, 2, 2]},
            "litevae-b": {"model_channels": 32, "ch_mult": [1, 2, 3]},
            "litevae-m": {"model_channels": 64, "ch_mult": [1, 2, 4]},
            "litevae-l": {"model_channels": 64, "ch_mult": [1, 2, 4]},
        }

        aggregator_configs = {
            "litevae-s": {"model_channels": 16, "ch_mult": [1, 2, 2]},
            "litevae-b": {"model_channels": 32, "ch_mult": [1, 2, 3]},
            "litevae-m": {"model_channels": 64, "ch_mult": [1, 2, 3]},
            "litevae-l": {"model_channels": 64, "ch_mult": [1, 2, 4]},
        }

        assert model_version in encoder_configs, f"Unknown model version: {model_version}"

        fe_cfg = encoder_configs[model_version]
        self.model_channels_extractor = fe_cfg["model_channels"]
        self.ch_mult_extractor        = fe_cfg["ch_mult"]

        fa_cfg = aggregator_configs[model_version]
        self.model_channels_aggregator = fa_cfg["model_channels"]
        self.ch_mult_aggregator        = fa_cfg["ch_mult"]

        # Shared UNets (1 per level)
        self.feature_extractor_L1 = LiteVAEUNetBlock(in_channels = self.in_channels, out_channels = self.out_channels,
                                                     model_channels = self.model_channels_extractor, ch_multiplies = self.ch_mult_extractor) # replacing in_channels with #sub-bands
        self.feature_extractor_L2 = LiteVAEUNetBlock(in_channels = self.in_channels, out_channels = self.out_channels,
                                                     model_channels = self.model_channels_extractor, ch_multiplies = self.ch_mult_extractor)
        self.feature_extractor_L3 = LiteVAEUNetBlock(in_channels = self.in_channels, out_channels = self.out_channels,
                                                     model_channels = self.model_channels_extractor, ch_multiplies = self.ch_mult_extractor)

        # Aggregator UNet: (L1 + L2 + L3) --> 2 * out_channels or 12 fixed by paper or 8 for LMM model in GMS consistency
        aggregated_channels  = in_channels * 3
        out_channels_agg     = 8  # [mu | logvar]
        self.feature_aggregator = LiteVAEUNetBlock(in_channels = aggregated_channels, out_channels = out_channels_agg,
                                                   model_channels = self.model_channels_aggregator, ch_multiplies = self.ch_mult_aggregator)

        # Downsamplers for L1 and L2 features to match L3 size
        self.downsample_L1 = Downsample2D(in_channels, scale_factor = 4)
        self.downsample_L2 = Downsample2D(in_channels, scale_factor = 2)

    def forward(self, image: Tensor) -> Tensor:
        # Get all sub-bands from DWT for each level
# fh
        dwt_L1 = self.wavelet_fn.dwt(image, level = 1) / 2 # (B, 12, H/2, W/2)
        dwt_L2 = self.wavelet_fn.dwt(image, level = 2) / 4 # (B, 12, H/4, W/4)
        dwt_L3 = self.wavelet_fn.dwt(image, level = 3) / 8 # (B, 12, H/8, W/8)

        # Feature extraction (shared UNet blocks)
        feat_L1 = self.downsample_L1(self.feature_extractor_L1(dwt_L1)) # (B, 12, H/8, H/8)
        feat_L2 = self.downsample_L2(self.feature_extractor_L2(dwt_L2)) # (B, 12, H/8, H/8)
        feat_L3 = self.feature_extractor_L3(dwt_L3)

        # Concatenate and aggregate to latent (mu + logvar)
        feat_cat = torch.cat([feat_L1, feat_L2, feat_L3], dim = 1) # (B, 36, H/8, W/8)

        latent_out = self.feature_aggregator(feat_cat) # (B, 8, H/8, W/8) --> [mu | logvar]

        # dwt_L2     = self.wavelet_fn.dwt(image, level = 2) / 4 # (B, 12, H/4, W/4)
        # feat_L2    = self.downsample_L2(self.feature_extractor_L2(dwt_L2)) # (B, 12, H/4, W/4)
        # latent_out = self.feature_aggregator(feat_L2) # (B, 8, H/4, W/4) --> [mu | logvar]

        return latent_out