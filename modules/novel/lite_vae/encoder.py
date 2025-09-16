# ------------------------------------------------------------------------------#
#
# File name                 : encoder.py
# Purpose                   : LiteVAE encoder with Haar-based multi-scale feature extraction
# Usage                     : See example in `main()`
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# Note                      : Adapted from [https://arxiv.org/pdf/2405.14477.pdf]
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch

from torch                              import nn, Tensor
from modules.novel.lite_vae.blocks import LiteVAEUNetBlock
from modules.novel.lite_vae.blocks.haar import HaarTransform
from modules.novel.lite_vae.utils import Downsample2D
# ------------------------------------------------------------------------------#


# ----------------------------- LiteVAE Encoder --------------------------------#
class LiteVAEEncoder(nn.Module):
    """
    LiteVAE encoder that performs:
      - Multi-level Haar DWT (3 levels) to extract frequency bands
      - Shared UNet-based feature extraction at each level
      - Downsampling to a common resolution
      - Aggregation UNet to produce Gaussian parameters [mu | logvar]

    Output shape:
        Tensor of shape (B, 2C, H, W) where 2C = [mu, logvar]
    """

    def __init__(
        self,
        model_version: str = "litevae",  # Options: litevae, litevae-b, litevae-m, litevae-l
        in_channels: int   = 12,         # From 3-level Haar on RGB → 3×4=12 channels
        out_channels: int  = 12,         # Feature extractor output channels
        wavelet_fn: HaarTransform = HaarTransform(),
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.wavelet_fn    = wavelet_fn
        self.in_channels   = in_channels
        self.out_channels  = out_channels

        # --------------------------- Model Configurations - Taken from paper --------------------------- #
        encoder_cfgs = {
            "litevae"  : {"model_channels": 16, "ch_mult": [1, 2, 2]},
            "litevae-b": {"model_channels": 32, "ch_mult": [1, 2, 3]},
            "litevae-m": {"model_channels": 64, "ch_mult": [1, 2, 4]},
            "litevae-l": {"model_channels": 64, "ch_mult": [1, 2, 4]},
        }
        aggregator_cfgs = {
            "litevae"  : {"model_channels": 16, "ch_mult": [1, 2, 2]},
            "litevae-b": {"model_channels": 32, "ch_mult": [1, 2, 3]},
            "litevae-m": {"model_channels": 64, "ch_mult": [1, 2, 3]},
            "litevae-l": {"model_channels": 64, "ch_mult": [1, 2, 4]},
        }

        if model_version not in encoder_cfgs:
            raise ValueError(f"Unknown model version: {model_version}")

        fe_cfg = encoder_cfgs[model_version]
        fa_cfg = aggregator_cfgs[model_version]

        # ------------------------- Feature Extractors -------------------------------#
        self.feature_extractor_L1 = LiteVAEUNetBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            model_channels = fe_cfg["model_channels"],
            ch_multiplies  = fe_cfg["ch_mult"],
        )
        self.feature_extractor_L2 = LiteVAEUNetBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            model_channels = fe_cfg["model_channels"],
            ch_multiplies  = fe_cfg["ch_mult"],
        )
        self.feature_extractor_L3 = LiteVAEUNetBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            model_channels = fe_cfg["model_channels"],
            ch_multiplies  = fe_cfg["ch_mult"],
        )

        # ------------------------- Aggregation UNet ---------------------------------#
        aggregated_channels  = in_channels * 3
        out_channels_agg     = 8  # [mu | logvar] for LMM compatibility
        self.feature_aggregator = LiteVAEUNetBlock(
            in_channels  = aggregated_channels,
            out_channels = out_channels_agg,
            model_channels = fa_cfg["model_channels"],
            ch_multiplies  = fa_cfg["ch_mult"],
        )

        # ------------------------- Downsamplers -------------------------------------#
        self.downsample_L1 = Downsample2D(in_channels, scale_factor = 4)
        self.downsample_L2 = Downsample2D(in_channels, scale_factor = 2)

    # -------------------------------------------------------------------------- #
    def forward(self, image: Tensor) -> Tensor:
        """Forward pass: apply DWT, extract features, aggregate to latent."""
        dwt_L1 = self.wavelet_fn.dwt(image, level = 1) / 2
        dwt_L2 = self.wavelet_fn.dwt(image, level = 2) / 4
        dwt_L3 = self.wavelet_fn.dwt(image, level = 3) / 8

        feat_L1 = self.downsample_L1(self.feature_extractor_L1(dwt_L1))
        feat_L2 = self.downsample_L2(self.feature_extractor_L2(dwt_L2))
        feat_L3 = self.feature_extractor_L3(dwt_L3)

        feat_cat   = torch.cat([feat_L1, feat_L2, feat_L3], dim=1)
        latent_out = self.feature_aggregator(feat_cat) # 8 channels for [mu | logvar]

        return latent_out


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    # Example run
    model = LiteVAEEncoder(model_version="litevae")
    dummy = torch.randn(2, 3, 224, 224)  # RGB input
    out   = model(dummy)
    print(f"Output shape: {out.shape}")