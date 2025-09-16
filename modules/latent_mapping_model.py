# ------------------------------------------------------------------------------#
#
# File name                 : latent_mapping_model.py
# Purpose                   : LMM: Residual + Spatial-Attention U-Net (deep supervision)
# Usage                     : from networks.latent_mapping_model import ResAttnUNet_DS
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 25, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import logging
import torch
import torch.nn                 as nn
import torch.nn.functional      as F

from einops                     import rearrange

# ------------------------------------------------------------------------------#
#                         Norm & Attention                                       #
# ------------------------------------------------------------------------------#
def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)

class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm   = Normalize(in_channels)
        self.q      = nn.Conv2d(in_channels, in_channels, 1)
        self.k      = nn.Conv2d(in_channels, in_channels, 1)
        self.v      = nn.Conv2d(in_channels, in_channels, 1)
        self.proj   = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h       = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)

        b, c, hgt, wid = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')

        attn = torch.einsum('bij,bjk->bik', q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=2)
        attn = rearrange(attn, 'b i j -> b j i')

        v    = rearrange(v, 'b c h w -> b c (h w)')
        hout = torch.einsum('bij,bjk->bik', v, attn)
        hout = rearrange(hout, 'b c (h w) -> b c h w', h=hgt)
        return x + self.proj(hout)

# ------------------------------------------------------------------------------#
#                         ResBlocks                                             #
# ------------------------------------------------------------------------------#

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky: bool = True):
        super().__init__()
        act1            = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        act2            = nn.PReLU() if leaky else nn.ReLU(inplace=True)

        self.conv1      = nn.Sequential(Normalize(in_channels),  act1,
                                        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        self.conv2      = nn.Sequential(Normalize(out_channels), act2,
                                        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True))

        self.skip       = nn.Identity() if in_channels == out_channels \
                          else nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(self.conv1(x))
        return self.skip(x) + h

class ResAttBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res    = ResBlock(in_channels, out_channels)
        self.attn   = SpatialSelfAttention(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.res(x))

# ------------------------------------------------------------------------------#
#                         LMM: ResAttnUNet_DS                                    #
# ------------------------------------------------------------------------------#
class ResAttnUNet_DS(nn.Module):
    """Residual + spatial self-attention U-Net-like mapper with deep supervision.

    Input:  z  ∈ ℝ[B, Cin,  H,  W]
    Output: dict(level3, level2, level1, out), each ∈ ℝ[B, Cout, H, W]
    """
    def __init__(self, in_channel: int = 4, out_channels: int = 4,
                 num_res_blocks: int = 2, ch: int = 32, ch_mult=(1, 2, 4, 4)) -> None:
        super().__init__()

        if isinstance(num_res_blocks, int):
            _ = len(ch_mult) * [num_res_blocks]  # kept for parity with older signatures

        self.inp       = nn.Conv2d(in_channel, ch, 3, 1, 1, bias=True)

        self.conv1_0   = ResAttBlock(ch,               ch * ch_mult[0])
        self.conv2_0   = ResAttBlock(ch * ch_mult[0],  ch * ch_mult[1])
        self.conv3_0   = ResAttBlock(ch * ch_mult[1],  ch * ch_mult[2])
        self.conv4_0   = ResAttBlock(ch * ch_mult[2],  ch * ch_mult[3])

        self.conv3_1   = ResAttBlock(ch * (ch_mult[2] + ch_mult[3]), ch * ch_mult[2])
        self.conv2_2   = ResAttBlock(ch * (ch_mult[1] + ch_mult[2]), ch * ch_mult[1])
        self.conv1_3   = ResAttBlock(ch * (ch_mult[0] + ch_mult[1]), ch * ch_mult[0])
        self.conv0_4   = ResAttBlock(ch * (1 + ch_mult[0]),          ch)

        self.convds3   = nn.Sequential(Normalize(ch * ch_mult[2]), nn.SiLU(),
                                       nn.Conv2d(ch * ch_mult[2], out_channels, 3, 1, 1, bias=False))
        self.convds2   = nn.Sequential(Normalize(ch * ch_mult[1]), nn.SiLU(),
                                       nn.Conv2d(ch * ch_mult[1], out_channels, 3, 1, 1, bias=False))
        self.convds1   = nn.Sequential(Normalize(ch * ch_mult[0]), nn.SiLU(),
                                       nn.Conv2d(ch * ch_mult[0], out_channels, 3, 1, 1, bias=False))
        self.convds0   = nn.Sequential(Normalize(ch),              nn.SiLU(),
                                       nn.Conv2d(ch, out_channels, 3, 1, 1, bias=False))

        self._init_weights()
        self._print_networks(verbose=False)

    def forward(self, x: torch.Tensor) -> dict:
        # encoder path
        x0   = self.inp(x)
        x1   = self.conv1_0(x0)
        x2   = self.conv2_0(x1)
        x3   = self.conv3_0(x2)
        x4   = self.conv4_0(x3)

        # decoder path with skips
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

    # ------------------------- utils ------------------------------------------#
    def _init_weights(self) -> None:
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
        logging.info("---------- LMM initialized -------------")
        num_params = sum(p.numel() for p in self.parameters())
        if verbose:
            logging.info(self)
        logging.info("Total number of parameters : %.3f M", num_params / 1e6)
        logging.info("----------------------------------------")