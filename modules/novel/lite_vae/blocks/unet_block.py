# ------------------------------------------------------------------------------#
#
# File name                 : unet_block.py
# Purpose                   : LiteVAEUNetBlock — hierarchical encoder/decoder block with skips
# Usage                     : from networks.novel.lite_vae.blocks.unet_block import LiteVAEUNetBlock
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__                import annotations

from typing                    import List
import torch
import torch.nn                as nn

from modules.novel.lite_vae.blocks.resblock  import ResBlock
from modules.novel.lite_vae.blocks.midblock  import MidBlock2D
# ------------------------------------------------------------------------------#


# ------------------------------ LiteVAEUNetBlock ------------------------------#
class LiteVAEUNetBlock(nn.Module):
    """
    UNet-style feature extractor with symmetric skip connections.
    """
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        model_channels   : int = 16,
        ch_multiplies    : List[int] = [1, 2, 2],
        num_res_blocks   : int = 2,
    ) -> None:
        super().__init__()

        self.in_layer    = nn.Conv2d(in_channels,  model_channels, kernel_size=3, padding=1)
        self.out_layer   = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

        # Encoder path
        channel          = model_channels
        in_channel_list  = [channel]
        enc_blocks       = []
        for ch_mult in ch_multiplies:
            for _ in range(num_res_blocks):
                block = ResBlock(in_channels=channel, out_channels=model_channels * ch_mult)
                enc_blocks.append(block)
                channel = model_channels * ch_mult
                in_channel_list.append(channel)
        self.encoder_blocks = nn.ModuleList(enc_blocks)

        # Middle
        self.mid_block  = MidBlock2D(in_channels=channel, out_channels=channel)

        # Decoder path
        dec_blocks      = []
        for ch_mult in reversed(ch_multiplies):
            for _ in range(num_res_blocks):
                skip_ch = in_channel_list.pop()
                dec_blocks.append(
                    ResBlock(in_channels=channel + skip_ch, out_channels=model_channels * ch_mult)
                )
                channel = model_channels * ch_mult
        self.decoder_blocks = nn.ModuleList(dec_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x             = self.in_layer(x)
        skip_features = [x]

        for enc in self.encoder_blocks:
            x = enc(x)
            skip_features.append(x)

        x = self.mid_block(x)

        for dec in self.decoder_blocks:
            skip = skip_features.pop()
            x    = torch.cat([x, skip], dim=1)
            x    = dec(x)

        return self.out_layer(x)
# ------------------------------------------------------------------------------#