# ------------------------------------------------------------------------------#
#
# File name                 : unet_block.py
# Purpose                   : Defines LiteVAE UNet block with encoder, mid, and 
#                             decoder paths using residual blocks.
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
from typing                 import List

import torch

from torch                                   import nn
from modules.novel.lite_vae.blocks.resblock  import ResBlock
from modules.novel.lite_vae.blocks.midblock  import MidBlock2D
# ------------------------------------------------------------------------------#


# ------------------------------ LiteVAE UNet Block ----------------------------#
class LiteVAEUNetBlock(nn.Module):
    """
    UNet-like block for LiteVAE with encoder, mid-block, and decoder paths.
    """
    def __init__(
        self,
        in_channels: int,                 # Input channels (e.g. 4)
        out_channels: int,                # Output channels
        model_channels: int = 16,         # Base model channels (e.g. 32, 64, 128)
        ch_multiplies: List[int] = [1, 2, 2],
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.in_layer   = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.out_layer  = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

        # ----------------------------- Encoder --------------------------------#
        channel             = model_channels
        in_channel_list     = [channel]
        self.encoder_blocks = []

        for ch_mult in ch_multiplies:
            for _ in range(num_res_blocks):
                block = ResBlock(
                    in_channels  = channel,
                    out_channels = model_channels * ch_mult,
                )
                self.encoder_blocks.append(block)
                channel = model_channels * ch_mult
                in_channel_list.append(channel)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        # ------------------------------ Middle --------------------------------#
        self.mid_block = MidBlock2D(
            in_channels  = channel,
            out_channels = channel,
        )

        # ------------------------------ Decoder --------------------------------#
        self.decoder_blocks = []
        for ch_mult in reversed(ch_multiplies):
            for _ in range(num_res_blocks):
                skip_channels = in_channel_list.pop()
                self.decoder_blocks.append(
                    ResBlock(
                        in_channels  = channel + skip_channels,
                        out_channels = model_channels * ch_mult,
                    )
                )
                channel = model_channels * ch_mult

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

    # --------------------------------------------------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x             = self.in_layer(x)
        skip_features = [x]

        # Encoder
        for enc_block in self.encoder_blocks:
            x = enc_block(x)
            skip_features.append(x)

        # Middle
        x = self.mid_block(x)

        # Decoder
        for dec_block in self.decoder_blocks:
            skip = skip_features.pop()
            x    = torch.cat([x, skip], dim=1)
            x    = dec_block(x)

        return self.out_layer(x)