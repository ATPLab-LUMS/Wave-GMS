# ------------------------------------------------------------------------------#
#
# File name                 : __init__.py
# Purpose                   : Package exports for tools/ (logging, ckpt I/O, losses,
#                             schedulers, metrics, general utilities)
# Usage                     : from tools import seed_reproducer, WeightedDiceBCE, ...
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
# Local utilities (explicit re-exports)
from .utils                          import (
    mkdir,
    seed_reproducer,
    load_checkpoint,
    save_checkpoint,
    count_params,
    adjust_learning_rate,
    get_cuda,
    print_options,
)

from .metrics                        import (
    paper_dice_iou,
    dice_score,
    iou_score,
    hd95_score,
    ssim,
    ssim_region,
    ssim_object,
    ssim_combined,
    all_metrics,
)

from .losses                         import (
    WeightedBCE,
    WeightedDiceLoss,
    WeightedDiceBCE,
)

from .lr_scheduler                   import (
    LinearWarmupCosineAnnealingLR,
)

# Keep legacy logger exports for compatibility with existing scripts.
# (If you later migrate to a different logger helper, update these re-exports.)
from .get_logger                     import (
    open_log,
    initLogging,
)

# Optional helpers for model loading / weights (surrounded by try for optional deps)
try:
    from .load_ckpt                  import (
        get_state_dict,
        get_tiny_autoencoder,
        get_lite_vae,
        get_sd_vae,
        get_imagenet_1k_litevae_pretrained,
        load_residual_tiny_vae,
    )
except Exception:  # pragma: no cover
    # Some environments may not have diffusers / HF hub available;
    # keep the package importable without failing.
    pass
# ------------------------------------------------------------------------------#

__all__ = [
    # utils
    "mkdir", "seed_reproducer", "load_checkpoint", "save_checkpoint",
    "count_params", "adjust_learning_rate", "get_cuda", "print_options",
    # metrics
    "paper_dice_iou", "dice_score", "iou_score", "hd95_score",
    "ssim", "ssim_region", "ssim_object", "ssim_combined", "all_metrics",
    # losses
    "WeightedBCE", "WeightedDiceLoss", "WeightedDiceBCE",
    # schedulers
    "LinearWarmupCosineAnnealingLR",
    # logging
    "open_log", "initLogging",
    # (optional) model/ckpt helpers
    "get_state_dict", "get_tiny_autoencoder", "get_lite_vae",
    "get_sd_vae", "get_imagenet_1k_litevae_pretrained", "load_residual_tiny_vae",
]
# ------------------------------------------------------------------------------#