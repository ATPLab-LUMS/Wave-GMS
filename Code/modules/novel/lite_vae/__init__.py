# ------------------------------------------------------------------------------#
#
# File name                 : __init__.py
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# -----------------------------------------------------------------------------
# networks/novel/litevae/__init__.py
# -----------------------------------------------------------------------------
# Convenient imports for LiteVAE components
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# networks/novel/litevae/__init__.py
# -----------------------------------------------------------------------------
# Re-export core LiteVAE components so users can simply do:
#     from networks.novel.litevae import LiteVAE, LiteVAEEncoder, load_pretrained_decoder
# -----------------------------------------------------------------------------

from .litevae import LiteVAE
from .encoder import LiteVAEEncoder
from .decoder import load_pretrained_decoder
from .utils   import Downsample2D, DiagonalGaussianDistribution
from .blocks  import (
    HaarTransform,
    ResBlock,
    ResBlockWithSMC,
    SMC,
    MidBlock2D,
    LiteVAEUNetBlock,
)

__all__ = [
    # high‑level
    "LiteVAE",
    "LiteVAEEncoder",
    "load_pretrained_decoder",
    # blocks / utils
    "HaarTransform",
    "ResBlock",
    "ResBlockWithSMC",
    "SMC",
    "MidBlock2D",
    "LiteVAEUNetBlock",
    "Downsample2D",
    "DiagonalGaussianDistribution",
]
