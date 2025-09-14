# ------------------------------------------------------------------------------#
#
# File name                 : model.py
# Purpose                   : Core building blocks for the diffusion UNet —
#                             sinusoidal embeddings, (cross‑)attention, ResNet
#                             blocks, up/down‑sampling, etc.
# Usage                     : Imported by AutoencoderKL / Diffusion UNet.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 23, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module imports ----------------------------------#
import math, torch

import torch.nn             as nn
import numpy                as np

from inspect                import isfunction
from einops                 import rearrange
from typing                 import Optional, Any

# Attempt to use memory‑efficient attention from xFormers
try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILABLE = True
except ImportError:
    XFORMERS_IS_AVAILABLE = False
    print("[model.py] xformers not found ⇒ falling back to PyTorch attention.")


# --------------------------- Sinusoidal embeddings ----------------------------#

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Return sinusoidal timestep embeddings (fp32 regardless of global precision).

    Args:
        timesteps      : 1-D tensor of shape (B,), integer timesteps
        embedding_dim  : Size of output embedding vector
    """
    assert timesteps.ndim == 1, "timesteps must be 1-D"

    half_dim  = embedding_dim // 2
    freq      = torch.exp(
                    torch.arange(half_dim, dtype=torch.float32,
                                 device=timesteps.device) *
                    -(math.log(10000.0) / (half_dim - 1))
                 )  # (half_dim,)

    emb = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)          # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)          # (B, 2·half_dim)

    if embedding_dim % 2 == 1:                                        # Zero‑pad if odd
        emb = nn.functional.pad(emb, (0, 1))
    return emb                                                        # (B, embedding_dim)


# --------------------------- Helper functions ---------------------------------#

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d() if isfunction(d) else d


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation (a.k.a. SiLU)."""
    return x * torch.sigmoid(x)


def Normalize(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm wrapper (eps=1e-6)."""
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)


# --------------------------- Cross‑attention (xFormers) ------------------------#

class MemoryEfficientCrossAttention(nn.Module):
    """Cross-attention layer that leverages xFormers memory-efficient kernels
    when available. Falls back to vanilla attention if xFormers is missing.
    """

    def __init__(
        self,
        query_dim   : int,
        context_dim : Optional[int] = None,
        heads       : int           = 8,
        dim_head    : int           = 64,
        dropout     : float         = 0.0,
    ) -> None:
        super().__init__()
        print(f"[ME-Attention] query_dim={query_dim} context_dim={context_dim} heads={heads}")

        inner_dim   = dim_head * heads
        context_dim = default(context_dim, query_dim)

        # Projections
        self.to_q = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.heads        = heads
        self.dim_head     = dim_head
        self.attention_op : Optional[Any] = None  # xformers kernel selector

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask=None):
        context = default(context, x)
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        b, _, _ = q.shape
        reshape = lambda t: (t.unsqueeze(3)
                               .reshape(b, t.shape[1], self.heads, self.dim_head)
                               .permute(0, 2, 1, 3)
                               .reshape(b * self.heads, t.shape[1], self.dim_head))
        q, k, v = map(reshape, (q, k, v))

        if XFORMERS_IS_AVAILABLE:
            out = xformers.ops.memory_efficient_attention(q, k, v, op=self.attention_op)
        else:  # Fallback — scaled dot‑product attention
            scale = self.dim_head ** -0.5
            attn  = (q @ k.transpose(1, 2)) * scale
            attn  = attn.softmax(dim=-1)
            out   = attn @ v

        out = (out.view(b, self.heads, -1, self.dim_head)
                   .permute(0, 2, 1, 3)
                   .reshape(b, -1, self.heads * self.dim_head))
        return self.to_out(out)


# --------------------------- Upsample block ------------------------------------#

class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


# --------------------------- Downsample block ----------------------------------#

class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = nn.functional.pad(x, (0, 1, 0, 1))  # explicit padding
            return self.conv(x)
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


# --------------------------- ResNet block --------------------------------------#

class ResnetBlock(nn.Module):
    """A time-conditional ResNet block used throughout the UNet backbone."""

    def __init__(
        self,
        *,
        in_channels     : int,
        out_channels    : Optional[int] = None,
        conv_shortcut   : bool          = False,
        dropout         : float         = 0.0,
        temb_channels   : int           = 512,
    ) -> None:
        super().__init__()
        out_channels              = default(out_channels, in_channels)
        self.in_channels          = in_channels
        self.out_channels         = out_channels
        self.use_conv_shortcut    = conv_shortcut

        # --- First conv -------------------------------------------------------#
        self.norm1  = Normalize(in_channels)
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Embed projection (if temb provided)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None

        # --- Second conv ------------------------------------------------------#
        self.norm2   = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # --- Shortcut ---------------------------------------------------------#
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut  = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]):
        h = self.conv1(nonlinearity(self.norm1(x)))

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb)).unsqueeze(-1).unsqueeze(-1)

        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


# --------------------------- Self‑Attention block ------------------------------#
class AttnBlock(nn.Module):
    """Channel-wise self-attention for spatial feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm      = Normalize(channels)
        self.q         = nn.Conv2d(channels, channels, kernel_size=1)
        self.k         = nn.Conv2d(channels, channels, kernel_size=1)
        self.v         = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out  = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)

        b, c, h_, w_ = q.shape
        q = q.reshape(b, c, h_ * w_).permute(0, 2, 1)          # (B, HW, C)
        k = k.reshape(b, c, h_ * w_)                           # (B, C, HW)

        attn = torch.bmm(q, k) * (c ** -0.5)                   # (B, HW, HW)
        attn = attn.softmax(dim=-1)

        v   = v.reshape(b, c, h_ * w_)
        attn = attn.permute(0, 2, 1)                           # align dims
        out = torch.bmm(v, attn).reshape(b, c, h_, w_)

        return x + self.proj_out(out)
    
# --------------------------- Memory‑efficient attention block ------------------#
# --------------------------- Memory‑efficient attention block ------------------#
class MemoryEfficientAttnBlock(nn.Module):
    """
    Memory-efficient self-attention block using xformers.
    Note: single-head self-attention implementation.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels   = in_channels

        self.norm          = Normalize(in_channels)
        self.q             = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k             = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v             = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out      = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.attention_op  = None  # Placeholder for memory-efficient attention backend

    def forward(self, x):
        h_ = self.norm(x)

        q  = self.q(h_)
        k  = self.k(h_)
        v  = self.v(h_)

        B, C, H, W = q.shape
        q, k, v    = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

        q, k, v    = map(lambda t: t.unsqueeze(3).reshape(B, t.shape[1], 1, C).permute(0, 2, 1, 3).reshape(B, t.shape[1], C).contiguous(), (q, k, v))

        out        = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out        = out.unsqueeze(0).reshape(B, 1, out.shape[1], C).permute(0, 2, 1, 3).reshape(B, out.shape[1], C)
        out        = rearrange(out, 'b (h w) c -> b c h w', b=B, h=H, w=W, c=C)

        return x + self.proj_out(out)


# --------------------------- Cross-attention wrapper ---------------------------#
class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    """
    Wraps memory-efficient cross-attention for 2D feature maps.
    """
    def forward(self, x, context=None, mask=None):
        b, c, h, w = x.shape
        x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
        out        = super().forward(x_reshaped, context=context, mask=mask)
        out        = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w, c=c)
        return x + out


# --------------------------- Attention block factory ---------------------------#
def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    """
    Dynamically creates attention block based on attn_type.
    """
    valid_types = ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"]
    assert attn_type in valid_types, f"attn_type {attn_type} unknown"

    if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"

    print(f"Making attention of type '{attn_type}' with {in_channels} in_channels")

    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)

    elif attn_type == "vanilla-xformers":
        print(f"Building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)

    elif attn_type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)

    elif attn_type == "none":
        return nn.Identity()

    else:
        raise NotImplementedError()

# --------------------------- Encoder Network ---------------------------#
class Encoder(nn.Module):
    """
    Hierarchical encoder used to map input images to latent Gaussian parameters.
    Applies progressive downsampling with ResNet blocks and attention.
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()

        if use_linear_attn:
            attn_type = "linear"

        self.ch               = ch
        self.temb_ch          = 0
        self.num_resolutions  = len(ch_mult)
        self.num_res_blocks   = num_res_blocks
        self.resolution       = resolution
        self.in_channels      = in_channels

        self.conv_in = torch.nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res     = resolution
        in_ch_mult   = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down     = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block     = nn.ModuleList()
            attn      = nn.ModuleList()
            block_in  = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            down = nn.Module()
            down.block = block
            down.attn  = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2

            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, self.temb_ch, dropout)
        self.mid.attn_1  = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(block_in, block_in, self.temb_ch, dropout)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        temb = None
        hs   = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = self.mid.block_1(hs[-1], temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# --------------------------- Decoder Network ---------------------------#
class Decoder(nn.Module):
    """
    Hierarchical decoder that reconstructs output image from latent representation.
    Mirrors the encoder with progressive upsampling.
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False,
                 use_linear_attn=False, attn_type="vanilla", **ignorekwargs):
        super().__init__()

        if use_linear_attn:
            attn_type = "linear"

        self.ch              = ch
        self.temb_ch         = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks  = num_res_blocks
        self.resolution      = resolution
        self.in_channels     = in_channels
        self.give_pre_end    = give_pre_end
        self.tanh_out        = tanh_out

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in   = ch * ch_mult[-1]
        curr_res   = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, self.temb_ch, dropout)
        self.mid.attn_1  = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(block_in, block_in, self.temb_ch, dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block     = nn.ModuleList()
            attn      = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, self.temb_ch, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))

            up = nn.Module()
            up.block = block
            up.attn  = attn

            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2

            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None

        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

# --------------------------------- End -----------------------------------------#
