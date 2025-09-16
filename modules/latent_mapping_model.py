# ------------------------------------------------------------------------------#
#
# File name                 : latent_mapping_model.py
# Purpose                   : Defines the Latent Mapping Model (LMM) with residual
#                             and attention-based blocks for latent-to-latent mapping.
# Usage                     : See example in main()
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# Note:                     : Adapted from [https://github.com/King-HAW/GMS]
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import logging

import torch
from torch import nn
from einops import rearrange
# ------------------------------------------------------------------------------#


# ----------------------------- Normalization ----------------------------------#
def Normalize(in_channels: int) -> nn.GroupNorm:
    """GroupNorm layer with fixed configuration."""
    return nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)


# --------------------------- Attention Block ----------------------------------#
class SpatialSelfAttention(nn.Module):
    """Spatial self-attention block with normalization and projection."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm       = Normalize(in_channels)
        self.q          = nn.Conv2d(in_channels, in_channels, 1)
        self.k          = nn.Conv2d(in_channels, in_channels, 1)
        self.v          = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out   = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        w_ = torch.einsum("bij,bjk->bik", q, k) * (c ** -0.5)
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v  = rearrange(v, "b c h w -> b c (h w)")
        w_ = rearrange(w_, "b i j -> b j i")
        h_ = torch.einsum("bij,bjk->bik", v, w_)
        h_ = rearrange(h_, "b c (h w) -> b c h w", h=h)
        return x + self.proj_out(h_)


# ---------------------------- Residual Blocks ---------------------------------#
class ResBlock(nn.Module):
    """Residual block with optional channel matching."""

    def __init__(self, in_channels: int, out_channels: int, leaky: bool = True) -> None:
        super().__init__()
        act1, act2 = (nn.PReLU(), nn.PReLU()) if leaky else (nn.ReLU(inplace=True), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(
            Normalize(in_channels),
            act1,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = nn.Sequential(
            Normalize(out_channels),
            act2,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_connection(x) + self.conv2(self.conv1(x))


class ResAttBlock(nn.Module):
    """Residual block followed by spatial self-attention."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.resblock  = ResBlock(in_channels, out_channels)
        self.attention = SpatialSelfAttention(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.resblock(x))


# ----------------------------- LMM (UNet) -------------------------------------#
class ResAttnUNet_DS(nn.Module):
    """Latent Mapping Model (ResAttn U-Net with deep supervision)."""

    def __init__(self, in_channel: int = 8, out_channels: int = 8,
                 num_res_blocks: int = 2, ch: int = 32,
                 ch_mult: tuple = (1, 2, 4, 4)) -> None:
        super().__init__()
        self.ch             = ch
        self.num_res_blocks = len(ch_mult) * [num_res_blocks] if isinstance(num_res_blocks, int) else num_res_blocks

        # Encoder
        self.input_blocks   = nn.Conv2d(in_channel, ch, kernel_size=3, stride=1, padding=1)
        self.conv1_0        = ResAttBlock(ch, ch * ch_mult[0])
        self.conv2_0        = ResAttBlock(ch * ch_mult[0], ch * ch_mult[1])
        self.conv3_0        = ResAttBlock(ch * ch_mult[1], ch * ch_mult[2])
        self.conv4_0        = ResAttBlock(ch * ch_mult[2], ch * ch_mult[3])

        # Decoder
        self.conv3_1        = ResAttBlock(ch * (ch_mult[2] + ch_mult[3]), ch * ch_mult[2])
        self.conv2_2        = ResAttBlock(ch * (ch_mult[1] + ch_mult[2]), ch * ch_mult[1])
        self.conv1_3        = ResAttBlock(ch * (ch_mult[0] + ch_mult[1]), ch * ch_mult[0])
        self.conv0_4        = ResAttBlock(ch * (1 + ch_mult[0]), ch)

        # Deep supervision heads
        self.convds3        = nn.Sequential(Normalize(ch * ch_mult[2]), nn.SiLU(),
                                            nn.Conv2d(ch * ch_mult[2], out_channels, 3, 1, 1, bias=False))
        self.convds2        = nn.Sequential(Normalize(ch * ch_mult[1]), nn.SiLU(),
                                            nn.Conv2d(ch * ch_mult[1], out_channels, 3, 1, 1, bias=False))
        self.convds1        = nn.Sequential(Normalize(ch * ch_mult[0]), nn.SiLU(),
                                            nn.Conv2d(ch * ch_mult[0], out_channels, 3, 1, 1, bias=False))
        self.convds0        = nn.Sequential(Normalize(ch), nn.SiLU(),
                                            nn.Conv2d(ch, out_channels, 3, 1, 1, bias=False))

        self._initialize_weights()
        self._print_networks(verbose=False)

    def forward(self, x: torch.Tensor) -> dict:
        x0  = self.input_blocks(x)
        x1  = self.conv1_0(x0)
        x2  = self.conv2_0(x1)
        x3  = self.conv3_0(x2)
        x4  = self.conv4_0(x3)

        x3_1 = self.conv3_1(torch.cat([x3, x4], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, x3_1], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, x2_2], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, x1_3], dim=1))

        return {
            "level3": self.convds3(x3_1),
            "level2": self.convds2(x2_2),
            "level1": self.convds1(x1_3),
            "out":    self.convds0(x0_4),
        }

    # --------------------------- He Initialization & Number of Parameters -------------------------------#
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _print_networks(self, verbose: bool = False) -> None:
        logging.info("---------- Networks initialized -------------")
        num_params = sum(p.numel() for p in self.parameters())
        if verbose:
            logging.info(self.modules())
        logging.info("Total number of parameters : %.3f M" % (num_params / 1e6))
        logging.info("-----------------------------------------------")


# ------------------------------ Main ------------------------------------------#
if __name__ == "__main__":
    # Example: Run standalone test
    model = ResAttnUNet_DS(in_channel=4, out_channels=4, num_res_blocks=2, ch=32, ch_mult=(1, 2, 4, 4))
    out_dict = model(torch.ones(2, 4, 64, 64))
    for key, val in out_dict.items():
        print(f"{key}, shape: {val.shape}")