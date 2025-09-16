# ------------------------------------------------------------------------------#
#
# File name                 : haar.py
# Purpose                   : Implements Haar wavelet transform (DWT/IDWT) module 
#                             for multi-resolution feature decomposition.
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
import torch
import torch.nn as nn

import pywt
import ptwt
# ------------------------------------------------------------------------------#


# ---------------------------- Haar Transform ----------------------------------#
class HaarTransform(nn.Module):
    """
    Haar wavelet transform module with both forward (DWT) and inverse (IDWT) operations.
    """
    def __init__(self, level: int = 3, mode: str = "symmetric", with_grad: bool = False) -> None:
        super().__init__()
        self.wavelet   = pywt.Wavelet("haar")
        self.level     = level
        self.mode      = mode
        self.with_grad = with_grad

    # ---------------------------- Forward DWT ---------------------------------#
    def dwt(self, x: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """
        Apply discrete wavelet transform (DWT).
        """
        with torch.set_grad_enabled(self.with_grad):
            level        = level or self.level
            x_low, *x_hi = ptwt.wavedec2(x.float(), wavelet=self.wavelet, level=level, mode=self.mode)
            return torch.cat([x_low, x_hi[0][0], x_hi[0][1], x_hi[0][2]], dim=1)

    # ---------------------------- Inverse DWT ---------------------------------#
    def idwt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse discrete wavelet transform (IDWT).
        """
        with torch.set_grad_enabled(self.with_grad):
            x_low   = x[:, :3]
            x_high  = torch.chunk(x[:, 3:], 3, dim=1)
            return ptwt.waverec2([x_low.float(), x_high], wavelet=self.wavelet)

    # ----------------------------- Forward Pass -------------------------------#
    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        return self.idwt(x) if inverse else self.dwt(x)