# ------------------------------------------------------------------------------#
#
# File name                 : haar.py
# Purpose                   : Haar DWT/IDWT wrapper using pywt/ptwt
# Usage                     : from networks.novel.lite_vae.blocks.haar import HaarTransform
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import torch
import torch.nn                   as nn
import pywt
import ptwt
# ------------------------------------------------------------------------------#


# ----------------------------- Haar Transform ---------------------------------#
class HaarTransform(nn.Module):
    """
    DWT/IDWT helper. Current behavior (kept for compatibility):
    - dwt(x, level=k) returns concat([x_low, H1, V1, D1]) i.e., only level-1 high bands.
      This matches your existing pipelines. Multi-level usage can be added later.
    """
    def __init__(self, level: int = 3, mode: str = "symmetric", with_grad: bool = False) -> None:
        super().__init__()
        self.wavelet     = pywt.Wavelet("haar")
        self.level       = level
        self.mode        = mode
        self.with_grad   = with_grad

    # ---------------------------------------- Forward DWT ---------------------------------------- #
    def dwt(self, x: torch.Tensor, level: int | None = None) -> torch.Tensor:
        with torch.set_grad_enabled(self.with_grad):
            level          = level or self.level
            x_low, *x_high = ptwt.wavedec2(x.float(), wavelet=self.wavelet, level=level, mode=self.mode)
            # Keep only level-1 high-frequency bands for compatibility
            x_combined     = torch.cat([x_low, x_high[0][0], x_high[0][1], x_high[0][2]], dim=1)
            return x_combined

    # ---------------------------------------- Inverse DWT ---------------------------------------- #
    def idwt(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.with_grad):
            x_low   = x[:, :3]
            x_high  = torch.chunk(x[:, 3:], 3, dim=1)
            x_recon = ptwt.waverec2([x_low.float(), x_high], wavelet=self.wavelet)
            return x_recon

    # ---------------------------------------- Forward Pass --------------------------------------- #
    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        return self.idwt(x) if inverse else self.dwt(x)
# ------------------------------------------------------------------------------#