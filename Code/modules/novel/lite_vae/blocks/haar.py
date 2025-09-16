# ------------------------------------------------------------------------------#
#
# File name                 : haar.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# ----------------------------------- Haar Transform Module ----------------------------------- #

import torch, pywt, ptwt
from torch import nn

class HaarTransform(nn.Module):
    def __init__(self, level=3, mode="symmetric", with_grad=False) -> None:
        super().__init__()
        self.wavelet     = pywt.Wavelet("haar")
        self.level       = level
        self.mode        = mode
        self.with_grad   = with_grad

    # ---------------------------------------- Forward DWT ---------------------------------------- #
    def dwt(self, x, level=None):
        with torch.set_grad_enabled(self.with_grad):
            level        = level or self.level
            x_low, *x_high = ptwt.wavedec2(
                x.float(),
                wavelet=self.wavelet,
                level=level,
                mode=self.mode,
            )
            x_combined = torch.cat(
                [x_low, x_high[0][0], x_high[0][1], x_high[0][2]], dim=1
            )
            return x_combined

    # ---------------------------------------- Inverse DWT ---------------------------------------- #
    def idwt(self, x):
        with torch.set_grad_enabled(self.with_grad):
            x_low       = x[:, :3]
            x_high      = torch.chunk(x[:, 3:], 3, dim=1)
            x_recon     = ptwt.waverec2([x_low.float(), x_high], wavelet=self.wavelet)
            return x_recon

    # ---------------------------------------- Forward Pass ---------------------------------------- #
    def forward(self, x, inverse=False):
        if inverse:
            return self.idwt(x)
        return self.dwt(x)