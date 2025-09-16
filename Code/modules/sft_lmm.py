# ------------------------------------------------------------------------------#
#
# File name                 : sft_lmm.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

import torch

import torch.nn            as nn
import torch.nn.functional as F

from einops import rearrange

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)

class SFT(nn.Module):
    def __init__(self, in_channels, guidance_channels, nhidden=128, ks=3):
        super().__init__()
        pw = ks // 2
        self.shared = nn.Sequential(
            nn.Conv2d(guidance_channels, nhidden, kernel_size=ks, stride=1, padding=pw),
            nn.LeakyReLU(0.2, True)
        )
        self.gamma = nn.Conv2d(nhidden, in_channels, kernel_size=ks, stride=1, padding=pw)
        self.beta = nn.Conv2d(nhidden, in_channels, kernel_size=ks, stride=1, padding=pw)

    def forward(self, x, g):
        # g = F.adaptive_avg_pool2d(g, x.size()[2:]) # since the guidance is now not neccessarily of the same spatial dimensons as our ZI. For dino with 392 spatial dimensions it matches 28 x 28
        s = self.shared(g)
        gamma = self.gamma(s)
        beta = self.beta(s)
        return x * (gamma) + beta

class SFTResblk(nn.Module):
    def __init__(self, original_channels, guidance_channels, ks=3, dtype = torch.float32, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.conv_0 = nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(original_channels, original_channels, kernel_size=3, padding=1)

        self.norm_0 = SFT(original_channels, guidance_channels, ks=ks).to(dtype = dtype, device = device)
        self.norm_1 = SFT(original_channels, guidance_channels, ks=ks).to(dtype = dtype, device = device)

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x, ref):
        dx = self.conv_0(self.actvn(self.norm_0(x, ref)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, ref)))
        out = x + dx

        return out

class SFTModule(nn.Module):
    def __init__(self, original_channels, guidance_channels, dtype = torch.float32, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.sftresblk = SFTResblk(original_channels, guidance_channels).to(dtype = torch.float32, device = device)

    def forward(self, x, ref):
        x = self.sftresblk(x, ref)
        return x


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, h, w = q.size()
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k) * (c ** -0.5)
        w_ = F.softmax(w_, dim=2)

        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_out = torch.einsum('bij,bjk->bik', v, w_)
        h_out = rearrange(h_out, 'b c (h w) -> b c h w', h=h)
        return x + self.proj(h_out)

class ResBlock(nn.Module):
    """
    Convolutional blocks
    """
    def __init__(self, in_channels, out_channels, leaky=True):
        super().__init__()
        # activation, support PReLU and common ReLU
        self.act1 = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.act2 = nn.PReLU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            Normalize(in_channels),
            self.act1,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )

        self.conv2 = nn.Sequential(
            Normalize(out_channels),
            self.act2,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.skip_connection(x) + h


class ResAttBlock(nn.Module):
    """
    Residual + optional SFT + spatial self-attention.
    If order == 0:  x -> ResBlock -> SFT -> Attn
    If order == 1:  x -> SFT -> ResBlock -> Attn
    If identity is True, SFT becomes Identity (no guidance used).
    """
    def __init__(self, in_channels, out_channels, guidance_channels, identity=False, order=0):
        super().__init__()
        assert order in (0, 1), "order must be 0 or 1"

        self.identity = identity
        self.order = order

        self.resblock = ResBlock(in_channels=in_channels, out_channels=out_channels)

        if identity:
            self.sft = nn.Identity()
            # Note: Identity() accepts only (x), so we'll branch in forward.
        else:
            if order == 0:
                # after resblock -> channels are out_channels
                self.sft = SFT(in_channels=out_channels, guidance_channels=guidance_channels)
            else:  # order == 1
                # before resblock -> channels are in_channels
                self.sft = SFT(in_channels=in_channels, guidance_channels=guidance_channels)

        self.attention = SpatialSelfAttention(out_channels)

    def forward(self, x, guidance):
        if self.order == 0:
            h = self.resblock(x)
            h = h if self.identity else self.sft(h, guidance)
            h = self.attention(h)
            return h
        else:  # order == 1
            h = x if self.identity else self.sft(x, guidance)
            h = self.resblock(h)
            h = self.attention(h)
            return h

class SFT_UNet_DS(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, ch=32, ch_mult=(1, 2, 4, 4), guidance_channels=64):
        super().__init__()
        self.ch = ch
        self.input_proj = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1_0 = ResAttBlock(ch, ch * ch_mult[0], guidance_channels, identity = False, order = 0)
        self.conv2_0 = ResAttBlock(ch * ch_mult[0], ch * ch_mult[1], guidance_channels, identity = False, order = 0)
        self.conv3_0 = ResAttBlock(ch * ch_mult[1], ch * ch_mult[2], guidance_channels, identity = False, order = 0)
        self.conv4_0 = ResAttBlock(ch * ch_mult[2], ch * ch_mult[3], guidance_channels, identity = False, order = 0)

        self.conv3_1 = ResAttBlock(ch * (ch_mult[2]+ch_mult[3]), ch * ch_mult[2], guidance_channels, identity = False, order = 0)
        self.conv2_2 = ResAttBlock(ch * (ch_mult[1]+ch_mult[2]), ch * ch_mult[1], guidance_channels, identity = False, order = 0)
        self.conv1_3 = ResAttBlock(ch * (ch_mult[0]+ch_mult[1]), ch * ch_mult[0], guidance_channels, identity = False, order = 0)
        self.conv0_4 = ResAttBlock(ch * (1 + ch_mult[0]), ch, guidance_channels, identity = False, order = 0)

        self.convds3 = nn.Sequential(Normalize(ch * ch_mult[2]), nn.SiLU(), nn.Conv2d(ch * ch_mult[2], out_channels, 3, 1, 1, bias=True))
        self.convds2 = nn.Sequential(Normalize(ch * ch_mult[1]), nn.SiLU(), nn.Conv2d(ch * ch_mult[1], out_channels, 3, 1, 1, bias=True))
        self.convds1 = nn.Sequential(Normalize(ch * ch_mult[0]), nn.SiLU(), nn.Conv2d(ch * ch_mult[0], out_channels, 3, 1, 1, bias=True))
        self.convds0 = nn.Sequential(Normalize(ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, 1, 1, bias=True))

        # self._initialize_weights()

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

    def forward(self, x, guidance, guidance_type='wavelet'):

        x0 = self.input_proj(x)
        x1 = self.conv1_0(x0, guidance)
        x2 = self.conv2_0(x1, guidance)
        x3 = self.conv3_0(x2, guidance)
        x4 = self.conv4_0(x3, guidance)

        x3_1 = self.conv3_1(torch.cat([x3, x4], dim=1), guidance)
        x2_2 = self.conv2_2(torch.cat([x2, x3_1], dim=1), guidance)
        x1_3 = self.conv1_3(torch.cat([x1, x2_2], dim=1), guidance)
        x0_4 = self.conv0_4(torch.cat([x0, x1_3], dim=1), guidance)

        level3 = self.convds3(x3_1)
        level2 = self.convds2(x2_2)
        level1 = self.convds1(x1_3)
        out    = self.convds0(x0_4)

        return {
            'level3': self.convds3(x3_1),
            'level2': self.convds2(x2_2),
            'level1': self.convds1(x1_3),
            'out':    self.convds0(x0_4)
        }


# Step 1: Inputs
# B = 2  # batch size
# ZI = torch.randn(B, 4, 28, 28)        # Latent from LiteVAE
# guidance = torch.randn(B, 3, 112, 112) # Guidance input (e.g., edge maps, wavelet bands, Dino features)
# # Step 2: Initialize the model
# model = SFT_UNet_DS(
#     in_channels   = 4,          # Matches ZI channels
#     out_channels = 4,        # Output latent for predicted mask
#     ch           = 32,                 # Base channel width
#     ch_mult      = (1, 2, 4, 4),  # Channel growth pattern
#     guidance_channels = 3   # Matches guidance input
# )
#
# # Step 3: Forward pass
# out_dict = model(ZI, guidance)
#
# # Step 4: Output breakdown
# for level, tensor in out_dict.items():
#     print(f"{level}: {tensor.shape}")
#
# from torchinfo import summary
# print(summary(model, input_size=[(B, 4, 28, 28), (B, 3, 112, 112)]))