# ------------------------------------------------------------------------------#
#
# File name                 : sft_lmm.py
# Purpose                   : Implements Spatial Feature Transform (SFT)-based 
#                             latent mapping model with attention and UNet structure.
# Usage                     : See example in main()
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# Note:                     : SFT/SFTResblk adapted from [https://arxiv.org/pdf/2404.18820.pdf]
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
import torch.nn               as nn
import torch.nn.functional    as F

from einops                   import rearrange
from modules.latent_mapping_model import Normalize, ResBlock, SpatialSelfAttention
# ------------------------------------------------------------------------------#


# ------------------------ Spatial Feature Transform ---------------------------#
class SFT(nn.Module):
    """
    Spatial Feature Transform (SFT) block that applies affine modulation 
    (gamma, beta) conditioned on guidance features.
    """
    def __init__(self, in_channels: int, guidance_channels: int, nhidden: int = 128, ks: int = 3):
        super().__init__()
        pw          = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(guidance_channels, nhidden, kernel_size=ks, stride=1, padding=pw),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.gamma  = nn.Conv2d(nhidden, in_channels, kernel_size=ks, stride=1, padding=pw)
        self.beta   = nn.Conv2d(nhidden, in_channels, kernel_size=ks, stride=1, padding=pw)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        s     = self.shared(g)
        gamma = self.gamma(s)
        beta  = self.beta(s)
        return x * gamma + beta


# -------------------------- SFT Residual Block --------------------------------#
class SFTResblk(nn.Module):
    """
    Residual block with two Conv layers modulated by SFT.
    """
    def __init__(
        self,
        original_channels: int,
        guidance_channels: int,
        ks: int = 3,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.conv_0 = nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1)

        self.norm_0 = SFT(original_channels, guidance_channels, ks=ks).to(dtype=dtype, device=device)
        self.norm_1 = SFT(original_channels, guidance_channels, ks=ks).to(dtype=dtype, device=device)

    def actvn(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 2e-1)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        dx  = self.conv_0(self.actvn(self.norm_0(x, ref)))
        dx  = self.conv_1(self.actvn(self.norm_1(dx, ref)))
        out = x + dx
        return out


# ----------------------------- SFT Module -------------------------------------#
class SFTModule(nn.Module):
    """
    Wrapper module around a single SFTResblk.
    """
    def __init__(
        self,
        original_channels: int,
        guidance_channels: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.sftresblk = SFTResblk(original_channels, guidance_channels).to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.sftresblk(x, ref)


# ------------------------- Residual + SFT + Attn ------------------------------#
class ResAttBlock(nn.Module):
    """
    Residual block with optional SFT and spatial self-attention.

    order=0: x -> ResBlock -> SFT -> Attn  
    order=1: x -> SFT -> ResBlock -> Attn
    """
    def __init__(self, in_channels: int, out_channels: int, guidance_channels: int, identity: bool = False, order: int = 0):
        super().__init__()
        assert order in (0, 1), "order must be 0 or 1"

        self.identity   = identity
        self.order      = order
        self.resblock   = ResBlock(in_channels=in_channels, out_channels=out_channels)

        if identity:
            self.sft = nn.Identity()
        else:
            if order == 0:
                self.sft = SFT(out_channels, guidance_channels)
            else:
                self.sft = SFT(in_channels, guidance_channels)

        self.attention  = SpatialSelfAttention(out_channels)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        if self.order == 0:
            h = self.resblock(x)
            h = h if self.identity else self.sft(h, guidance)
        else:
            h = x if self.identity else self.sft(x, guidance)
            h = self.resblock(h)
        return self.attention(h)


# --------------------------- SFT UNet with DS ---------------------------------#
class SFT_UNet_DS(nn.Module):
    """
    UNet-like architecture with SFT-modulated residual-attention blocks and 
    deep supervision outputs at multiple levels.
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4, ch: int = 32, ch_mult=(1, 2, 4, 4), guidance_channels: int = 64):
        super().__init__()
        self.ch         = ch
        self.input_proj = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1_0    = ResAttBlock(ch, ch * ch_mult[0], guidance_channels)
        self.conv2_0    = ResAttBlock(ch * ch_mult[0], ch * ch_mult[1], guidance_channels)
        self.conv3_0    = ResAttBlock(ch * ch_mult[1], ch * ch_mult[2], guidance_channels)
        self.conv4_0    = ResAttBlock(ch * ch_mult[2], ch * ch_mult[3], guidance_channels)

        self.conv3_1    = ResAttBlock(ch * (ch_mult[2] + ch_mult[3]), ch * ch_mult[2], guidance_channels)
        self.conv2_2    = ResAttBlock(ch * (ch_mult[1] + ch_mult[2]), ch * ch_mult[1], guidance_channels)
        self.conv1_3    = ResAttBlock(ch * (ch_mult[0] + ch_mult[1]), ch * ch_mult[0], guidance_channels)
        self.conv0_4    = ResAttBlock(ch * (1 + ch_mult[0]), ch, guidance_channels)

        self.convds3    = nn.Sequential(Normalize(ch * ch_mult[2]), nn.SiLU(), nn.Conv2d(ch * ch_mult[2], out_channels, 3, 1, 1, bias=True))
        self.convds2    = nn.Sequential(Normalize(ch * ch_mult[1]), nn.SiLU(), nn.Conv2d(ch * ch_mult[1], out_channels, 3, 1, 1, bias=True))
        self.convds1    = nn.Sequential(Normalize(ch * ch_mult[0]), nn.SiLU(), nn.Conv2d(ch * ch_mult[0], out_channels, 3, 1, 1, bias=True))
        self.convds0    = nn.Sequential(Normalize(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, 1, 1, bias=True))

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> dict:
        x0    = self.input_proj(x)
        x1    = self.conv1_0(x0, guidance)
        x2    = self.conv2_0(x1, guidance)
        x3    = self.conv3_0(x2, guidance)
        x4    = self.conv4_0(x3, guidance)

        x3_1  = self.conv3_1(torch.cat([x3, x4], dim=1), guidance)
        x2_2  = self.conv2_2(torch.cat([x2, x3_1], dim=1), guidance)
        x1_3  = self.conv1_3(torch.cat([x1, x2_2], dim=1), guidance)
        x0_4  = self.conv0_4(torch.cat([x0, x1_3], dim=1), guidance)

        return {
            "level3": self.convds3(x3_1),
            "level2": self.convds2(x2_2),
            "level1": self.convds1(x1_3),
            "out":    self.convds0(x0_4),
        }