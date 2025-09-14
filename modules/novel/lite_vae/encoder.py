# ------------------------------------------------------------------------------#
#
# File name                 : encoder.py
# Purpose                   : LiteVAEEncoder — multi-level DWT + shared UNet extractors + aggregator
# Usage                     : from networks.novel.lite_vae.encoder import LiteVAEEncoder
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

import torch
import torch.nn                              as nn
from torch                                   import Tensor

from modules.novel.lite_vae.blocks.unet_block import LiteVAEUNetBlock
from modules.novel.lite_vae.blocks.haar       import HaarTransform
from modules.novel.lite_vae.utils             import Downsample2D
# ------------------------------------------------------------------------------#


# ------------------------------- LiteVAEEncoder -------------------------------#
class LiteVAEEncoder(nn.Module):
    """
    3-level DWT → per-level UNet feature extraction → align to L3 → concat → aggregator UNet.
    Returns [mu|logvar] with 8 channels by default (μ:4, logσ²:4).
    """
    def __init__(
        self,
        model_version : str = "litevae-s",     # "litevae-s" | "litevae-b" | "litevae-m" | "litevae-l"
        in_channels   : int = 12,              # 3 (RGB) × 4 (low+HVD) per level
        out_channels  : int = 12,              # per-level extractor output channels
        wavelet_fn    : HaarTransform = HaarTransform(),
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.wavelet_fn    = wavelet_fn
        self.in_channels   = in_channels
        self.out_channels  = out_channels

        # Feature extractor configs
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

        fe = encoder_configs[model_version]
        fa = aggregator_configs[model_version]

        # Shared per-level UNets
        self.feat_L1 = LiteVAEUNetBlock(in_channels=self.in_channels, out_channels=self.out_channels,
                                        model_channels=fe["model_channels"], ch_multiplies=fe["ch_mult"])
        self.feat_L2 = LiteVAEUNetBlock(in_channels=self.in_channels, out_channels=self.out_channels,
                                        model_channels=fe["model_channels"], ch_multiplies=fe["ch_mult"])
        self.feat_L3 = LiteVAEUNetBlock(in_channels=self.in_channels, out_channels=self.out_channels,
                                        model_channels=fe["model_channels"], ch_multiplies=fe["ch_mult"])

        # Aggregator: (L1 + L2 + L3) → [mu|logvar] (8 channels for LMM)
        self.agg_in_ch  = in_channels * 3
        self.agg_out_ch = 8
        self.agg        = LiteVAEUNetBlock(in_channels=self.agg_in_ch, out_channels=self.agg_out_ch,
                                           model_channels=fa["model_channels"], ch_multiplies=fa["ch_mult"])

        # Downsample to L3 spatial size
        self.down_L1 = Downsample2D(in_channels, scale_factor=4)
        self.down_L2 = Downsample2D(in_channels, scale_factor=2)

    def forward(self, image: Tensor) -> Tensor:
        # DWTs — keep behavior identical to your current implementation
        dwt_L1 = self.wavelet_fn.dwt(image, level=1) / 2.0   # (B,12,H/2, W/2)
        dwt_L2 = self.wavelet_fn.dwt(image, level=2) / 4.0   # (B,12,H/4, W/4)
        dwt_L3 = self.wavelet_fn.dwt(image, level=3) / 8.0   # (B,12,H/8, W/8)

        # Per-level extraction
        f1 = self.down_L1(self.feat_L1(dwt_L1))              # → (B,12,H/8,W/8)
        f2 = self.down_L2(self.feat_L2(dwt_L2))              # → (B,12,H/8,W/8)
        f3 = self.feat_L3(dwt_L3)                            # → (B,12,H/8,W/8)

        # Concatenate & aggregate to [mu|logvar] (8 channels)
        feats   = torch.cat([f1, f2, f3], dim=1)             # (B,36,H/8,W/8)
        moments = self.agg(feats)                            # (B,8,H/8,W/8)
        return moments
# ------------------------------------------------------------------------------#