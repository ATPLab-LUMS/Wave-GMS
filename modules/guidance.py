# ------------------------------------------------------------------------------#
#
# File name                 : guidance.py
# Purpose                   : Provides Haar wavelet-based guidance and SKFF module
#                             for multi-resolution feature selection.
# Usage                     : See example in main()
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, 
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : September 17, 2025
# Note:                     : SKFF module self adapted from [https://arxiv.org/pdf/2003.06792.pdf]
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
import torch.nn               as nn
import torch.nn.functional    as F

from modules.novel.litevae.blocks.haar import HaarTransform
# ------------------------------------------------------------------------------#


# -------------------------- Wavelet Sub-bands ---------------------------------#
def get_wavelet_subbands(images: torch.Tensor, lv: int = 3) -> torch.Tensor:
    """
    Perform multi-level Haar DWT and return sub-bands without the approximation coeff.
    
    Args:
        images (torch.Tensor): Input tensor of shape (B, C, H, W).
        lv (int): Levels of wavelet decomposition.
    """
    wavelet_fn   = HaarTransform()
    sub_bands    = wavelet_fn.dwt(images, level=lv) / (2 ** lv)  # (B, 12, H/2, W/2)
    sub_bands    = sub_bands[:, 3:, :, :]                        # Drop approximation
    return sub_bands


# --------------------------- Guidance Wrapper ---------------------------------#
def prepare_guidance(image: torch.Tensor, mode: str = "wavelet") -> torch.Tensor:
    """
    Wrapper to select guidance type.
    
    Args:
        image (torch.Tensor): Input tensor.
        mode (str): Guidance type (only 'wavelet' supported).
    """
    if mode == "wavelet":
        return get_wavelet_subbands(image)
    raise NotImplementedError(f"Guidance mode '{mode}' is not implemented.")


# ---------------------- Selective Kernel Feature Fusion -----------------------#
class SKFF(nn.Module):
    """
    Selective Kernel Feature Fusion (SKFF) module for combining 
    horizontal, vertical, and diagonal wavelet edges.
    """

    def __init__(self, channels: int = 3, reduction: int = 8) -> None:
        super().__init__()
        reduced_channels   = max(1, channels // reduction)

        self.global_pool   = nn.AdaptiveAvgPool2d(1)
        self.reduce_conv   = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.PReLU() 
        )

        self.expand_conv1  = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.expand_conv2  = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)
        self.expand_conv3  = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False)

    def forward(self, wavelet_edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wavelet_edges (torch.Tensor): Input tensor (B, 3*C, H, W), where C is 
                                          channels per stream (horizontal, vertical, diagonal).
        Returns:
            torch.Tensor: Fused tensor of shape (B, C, H, W).
        """
        B, total_C, H, W = wavelet_edges.size()
        C                = total_C // 3

        wavelet_h        = wavelet_edges[:, 0:C, :, :]
        wavelet_v        = wavelet_edges[:, C:2*C, :, :]
        wavelet_d        = wavelet_edges[:, 2*C:3*C, :, :]

        fused            = wavelet_h + wavelet_v + wavelet_d
        z                = self.reduce_conv(self.global_pool(fused))

        v1, v2, v3       = self.expand_conv1(z), self.expand_conv2(z), self.expand_conv3(z)
        scores           = torch.stack([v1, v2, v3], dim=1)         # (B, 3, C, 1, 1)
        weights          = F.softmax(scores, dim=1)

        s1, s2, s3       = weights[:, 0], weights[:, 1], weights[:, 2]
        out              = s1 * wavelet_h + s2 * wavelet_v + s3 * wavelet_d
        return out