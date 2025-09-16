# ------------------------------------------------------------------------------#
#
# File name                 : unet_block.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/blocks/unet_block.py
# -----------------------------------------------------------------------------
# LiteVAEUNetBlock — hierarchical feature extractor for multi-level wavelet input.
# Used both in:
#     1) Per-level wavelet feature extraction (shared architecture)
#     2) Final feature aggregation after concatenation
# -----------------------------------------------------------------------------
# Example
# -------
# >>> from networks.novel.litevae.blocks.unet_block import LiteVAEUNetBlock
# >>> block = LiteVAEUNetBlock(3, 16, 32)
# >>> x = torch.randn(2, 3, 28, 28)
# >>> out = block(x)
# >>> out.shape
# torch.Size([2, 16, 28, 28])
# -----------------------------------------------------------------------------

import torch
from __future__                              import annotations
from typing                                  import List
from torch                                   import nn
from networks.novel.lite_vae.blocks.midblock import MidBlock2D
from networks.novel.lite_vae.blocks.resblock import (
    ResBlock,
    ResBlockWithSMC,
)

__all__ = ["LiteVAEUNetBlock"]


# -----------------------------------------------------------------------------
# Main UNet block
# -----------------------------------------------------------------------------
class LiteVAEUNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int, # to be 4
        out_channels: int, 
        model_channels: int = 16, # [32, 64, 128]
        ch_multiplies: List[int] = [1, 2, 2],
        num_res_blocks: int = 2,
        use_smc: bool = False,
    ) -> None:
        super().__init__()

        self.in_layer  = nn.Conv2d(in_channels, model_channels, kernel_size  = 3, padding = 1)
        self.out_layer = nn.Conv2d(model_channels, out_channels, kernel_size = 3, padding = 1)

        resblock_class = ResBlockWithSMC if use_smc else ResBlock

        # -----------------------------------------------------------------
        # Encoder path
        # -----------------------------------------------------------------
        channel             = model_channels
        in_channel_list     = [channel]
        self.encoder_blocks = [] 
        for ch_mult in ch_multiplies:
            for _ in range(num_res_blocks):
                block = resblock_class(
                    in_channels  = channel,
                    out_channels = model_channels * ch_mult,
                )
                self.encoder_blocks.append(block)
                channel = model_channels * ch_mult
                in_channel_list.append(channel)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        # -----------------------------------------------------------------
        # Middle block
        # -----------------------------------------------------------------
        self.mid_block = MidBlock2D(
            in_channels  = channel,
            out_channels = channel,
            use_smc      = use_smc,
        )

        # -----------------------------------------------------------------
        # Decoder path
        # -----------------------------------------------------------------
        self.decoder_blocks = []
        for ch_mult in reversed(ch_multiplies):
            for _ in range(num_res_blocks):
                skip_channels = in_channel_list.pop()
                self.decoder_blocks.append(
                    resblock_class(
                        in_channels  = channel + skip_channels,
                        out_channels = model_channels * ch_mult,
                    )
                )
                channel = model_channels * ch_mult
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x                = self.in_layer(x)
        skip_features    = [x]

        for enc_block in self.encoder_blocks:
            x = enc_block(x)
            skip_features.append(x)

        x = self.mid_block(x)

        for dec_block in self.decoder_blocks:
            skip = skip_features.pop()
            x = torch.cat([x, skip], dim = 1)
            x = dec_block(x)

        return self.out_layer(x)