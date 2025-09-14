# ------------------------------------------------------------------------------#
#
# File name                 : sft_lmm.py
# Purpose                   : SFT-conditioned LMM (deep supervision) with safe guidance alignment
# Usage                     : from networks.sft_lmm import SFT_UNet_DS
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 25, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
import torch.nn                 as nn
import torch.nn.functional      as F
from einops                     import rearrange

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
#                         ResBlocks                                              #
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

# ------------------------------------------------------------------------------#
#                                    SFT                                        #
# ------------------------------------------------------------------------------#
class SFT(nn.Module):
    def __init__(self, in_channels: int, guidance_channels: int, nhidden: int = 128, ks: int = 3,
                 align_to_x: bool = True):
        
        super().__init__()
        pw          = ks // 2
        self.align  = align_to_x
        self.shared = nn.Sequential(
            nn.Conv2d(guidance_channels, nhidden, kernel_size=ks, stride=1, padding=pw),
            nn.PReLU()
            # nn.LeakyReLU(0.2, inplace=True),
        )
        self.gamma  = nn.Conv2d(nhidden, in_channels, kernel_size=ks, stride=1, padding=pw)
        self.beta   = nn.Conv2d(nhidden, in_channels, kernel_size=ks, stride=1, padding=pw)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.align and (g.shape[-2:] != x.shape[-2:]):
            # bilinear upsample/downsample to match x spatial size
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        s     = self.shared(g)
        gamma = self.gamma(s)
        beta  = self.beta(s)
        return x * gamma + beta

class SFTResblk(nn.Module):
    def __init__(self, original_channels: int, guidance_channels: int, ks: int = 3,
                 dtype: torch.dtype = torch.float32, device: str = None):
        super().__init__()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.conv0  = nn.Conv2d(original_channels, original_channels, 3, 1, 1)
        self.conv1  = nn.Conv2d(original_channels, original_channels, 3, 1, 1)

        self.norm0  = SFT(original_channels, guidance_channels, ks=ks).to(dtype=dtype, device=device)
        self.norm1  = SFT(original_channels, guidance_channels, ks=ks).to(dtype=dtype, device=device)

    def actvn(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 2e-1)

    def forward(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        dx  = self.conv0(self.actvn(self.norm0(x, ref)))
        dx  = self.conv1(self.actvn(self.norm1(dx, ref)))
        out = x + dx
        return out
    
# ------------------------------------------------------------------------------#
#                         Res/SFT/Att Blocks                                    #
# ------------------------------------------------------------------------------#

class ResAttBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, guidance_channels: int):
        super().__init__()
        self.res    = ResBlock(in_channels, out_channels)
        self.sft    = SFT(out_channels, guidance_channels)
        self.attn   = SpatialSelfAttention(out_channels)

    def forward(self, x: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        h = self.res(x)
        h = self.sft(h, guidance)
        h = self.attn(h)
        return h

# ------------------------------------------------------------------------------#
#                         SFT-Conditioned LMM                                    #
# ------------------------------------------------------------------------------#
class SFT_UNet_DS(nn.Module):
    """SFT-conditioned LMM with deep supervision heads.

    Input:  z        ∈ ℝ[B, Cin,  H,  W]
            guidance ∈ ℝ[B, Cg,  Hg, Wg] (auto-aligned to H×W)
    Output: dict(level3, level2, level1, out), each ∈ ℝ[B, Cout, H, W]
    """
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 ch: int = 32, ch_mult=(1, 2, 4, 4), guidance_channels: int = 64):
        super().__init__()

        self.ch         = ch
        self.input_proj = nn.Conv2d(in_channels, ch, 3, 1, 1, bias = True)

        self.conv1_0    = ResAttBlock(ch,               ch * ch_mult[0], guidance_channels)
        self.conv2_0    = ResAttBlock(ch * ch_mult[0],  ch * ch_mult[1], guidance_channels)
        self.conv3_0    = ResAttBlock(ch * ch_mult[1],  ch * ch_mult[2], guidance_channels)
        self.conv4_0    = ResAttBlock(ch * ch_mult[2],  ch * ch_mult[3], guidance_channels)

        self.conv3_1    = ResAttBlock(ch * (ch_mult[2] + ch_mult[3]), ch * ch_mult[2], guidance_channels)
        self.conv2_2    = ResAttBlock(ch * (ch_mult[1] + ch_mult[2]), ch * ch_mult[1], guidance_channels)
        self.conv1_3    = ResAttBlock(ch * (ch_mult[0] + ch_mult[1]), ch * ch_mult[0], guidance_channels)
        self.conv0_4    = ResAttBlock(ch * (1 + ch_mult[0]),          ch,              guidance_channels)

        self.convds3    = nn.Sequential(Normalize(ch * ch_mult[2]), nn.SiLU(),
                                        nn.Conv2d(ch * ch_mult[2], out_channels, 3, 1, 1, bias = False))
        self.convds2    = nn.Sequential(Normalize(ch * ch_mult[1]), nn.SiLU(),
                                        nn.Conv2d(ch * ch_mult[1], out_channels, 3, 1, 1, bias = False))
        self.convds1    = nn.Sequential(Normalize(ch * ch_mult[0]), nn.SiLU(),
                                        nn.Conv2d(ch * ch_mult[0], out_channels, 3, 1, 1, bias = False))
        self.convds0    = nn.Sequential(Normalize(ch),              nn.SiLU(),
                                        nn.Conv2d(ch, out_channels, 3, 1, 1, bias = False))
        
        self._init_weights()

    def forward(self, x: torch.Tensor, guidance: torch.Tensor, guidance_type: str = 'wavelet') -> dict:
        x0   = self.input_proj(x)
        x1   = self.conv1_0(x0, guidance)
        x2   = self.conv2_0(x1, guidance)
        x3   = self.conv3_0(x2, guidance)
        x4   = self.conv4_0(x3, guidance)

        x3_1 = self.conv3_1(torch.cat([x3, x4], dim=1), guidance)
        x2_2 = self.conv2_2(torch.cat([x2, x3_1], dim=1), guidance)
        x1_3 = self.conv1_3(torch.cat([x1, x2_2], dim=1), guidance)
        x0_4 = self.conv0_4(torch.cat([x0, x1_3], dim=1), guidance)

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

    # Main usage
if __name__ == "__main__":
    model = SFT_UNet_DS(in_channels=4, out_channels=4, ch=32, ch_mult=(1,2,4,4), guidance_channels=3)
    from torchinfo import summary

    print(summary(model, input_data=[torch.randn(1,4,28,28), torch.randn(1,3,112,112)], col_names=["input_size", "output_size", "num_params"]))