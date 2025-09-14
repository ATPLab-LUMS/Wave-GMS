# ------------------------------------------------------------------------------#
#
# File name                 : guidance_provider.py
# Purpose                   : Convenience wrapper to produce guidance tensors (edge/wavelet/dino)
# Usage                     : gp = GuidanceProvider(mode='dino'); g = gp(images, target_hw=(28,28))
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
import torch.nn.functional            as F
from   typing                         import Tuple

from modules.guidance                 import prepare_guidance

# ------------------------------------------------------------------------------#
#                         Provider                                               #
# ------------------------------------------------------------------------------#
class GuidanceProvider:
    def __init__(self, mode: str = "edge", align_to_hw: bool = True):
        """
        mode: 'edge' | 'wavelet' | 'dino'
        align_to_hw: if True, resize to target_hw in __call__
        """
        self.mode        = mode
        self.align_to_hw = align_to_hw

    def __call__(self, images: torch.Tensor, target_hw: Tuple[int,int] | None = None) -> torch.Tensor:
        g = prepare_guidance(images, mode=self.mode).to(device=images.device, dtype=images.dtype)
        if self.align_to_hw and target_hw is not None and g.shape[-2:] != target_hw:
            g = F.interpolate(g, size=target_hw, mode="bilinear", align_corners=False)
        return g
    
# other prompt

# ------------------------------------------------------------------------------#
#
# File name                 : guidance_provider.py
# Purpose                   : Convenience wrapper to produce guidance tensors
# Usage                     : g = GuidanceProvider("dino")(images, target_hw=(28,28))
#
# ------------------------------------------------------------------------------#

import torch
import torch.nn.functional as F
from typing import Tuple
from modules.guidance import prepare_guidance

class GuidanceProvider:
    def __init__(self, mode: str = "edge", align_to_hw: bool = True):
        self.mode        = mode
        self.align_to_hw = align_to_hw

    def __call__(self, images: torch.Tensor, target_hw: Tuple[int,int] | None = None) -> torch.Tensor:
        g = prepare_guidance(images, mode=self.mode).to(device=images.device, dtype=images.dtype)
        if self.align_to_hw and target_hw is not None and g.shape[-2:] != target_hw:
            g = F.interpolate(g, size=target_hw, mode="bilinear", align_corners=False)
        return g