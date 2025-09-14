# ------------------------------------------------------------------------------#
#
# File name                 : lr_scheduler.py
# Purpose                   : Linear warmup + cosine annealing LR schedulers (epoch- and step-based)
# Usage                     : from tools.lr_scheduler import (
#                               LinearWarmupCosineAnnealingLR,
#                               LinearWarmupCosineAnnealingLRSteps,
#                             )
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
import math
import warnings
from typing                      import List

from torch.optim                 import Optimizer
from torch.optim.lr_scheduler    import _LRScheduler
# ------------------------------------------------------------------------------#

__all__ = [
    "LinearWarmupCosineAnnealingLR",
    "LinearWarmupCosineAnnealingLRSteps",
]


# ======================= Epoch-based (original) ===============================#
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Epoch-based LR scheduler with a linear warmup followed by cosine annealing.

    Notes
    -----
    * Call `scheduler.step()` once per **epoch**.
    * If you pass an explicit epoch to `scheduler.step(epoch)`,
      PyTorch uses `_get_closed_form_lr()`.

    Args
    ----
    optimizer       : Wrapped optimizer.
    warmup_epochs   : Number of warmup epochs (>= 0).
    max_epochs      : Total epochs (warmup + cosine), must be > 0.
    warmup_start_lr : Starting LR at epoch 0 during warmup.
    eta_min         : Final LR at the end of cosine.
    last_epoch      : Index of last epoch; use -1 to start fresh.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if max_epochs <= 0:
            raise ValueError("max_epochs must be > 0")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if warmup_epochs > max_epochs:
            raise ValueError("warmup_epochs must be <= max_epochs")

        self.warmup_epochs    = int(warmup_epochs)
        self.max_epochs       = int(max_epochs)
        self.warmup_start_lr  = float(warmup_start_lr)
        self.eta_min          = float(eta_min)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last computed learning rate, call `get_last_lr()` instead.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        if 0 < self.last_epoch < self.warmup_epochs:
            # progress per warmup epoch (avoid /0)
            denom = max(self.warmup_epochs - 1, 1)
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / denom
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        if self.last_epoch == self.warmup_epochs:
            return list(self.base_lrs)

        # Cosine phase
        T_cur   = self.last_epoch - self.warmup_epochs
        T_total = max(self.max_epochs - self.warmup_epochs, 1)

        if (self.last_epoch - 1 - self.max_epochs) % (2 * T_total) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / T_total)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_total))
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_epochs:
            if self.last_epoch == 0:
                return [self.warmup_start_lr] * len(self.base_lrs)
            # linear interp from warmup_start_lr → base_lr
            denom = max(self.warmup_epochs, 1)
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / denom
                for base_lr in self.base_lrs
            ]

        T_cur   = self.last_epoch - self.warmup_epochs
        T_total = max(self.max_epochs - self.warmup_epochs, 1)

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_total))
            for base_lr in self.base_lrs
        ]


# ========================= Step-based (new) ===================================#
class LinearWarmupCosineAnnealingLRSteps(_LRScheduler):
    """
    **Step-based** LR scheduler with linear warmup (by steps/iterations) then cosine.

    Use this when you call `scheduler.step()` **each iteration**.

    Args
    ----
    optimizer       : Wrapped optimizer.
    warmup_steps    : Number of warmup steps (>= 0).
    max_steps       : Total steps (warmup + cosine), must be > 0.
    warmup_start_lr : Starting LR at step 0 during warmup.
    eta_min         : Final LR at the end of cosine.
    last_step       : Index of last step; use -1 to start fresh.

    Notes
    -----
    * This mirrors the epoch-based variant but counts **steps**.
    * You can still call `scheduler.step(step_idx)` with an explicit step to
      trigger `_get_closed_form_lr()`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_step: int = -1,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if warmup_steps > max_steps:
            raise ValueError("warmup_steps must be <= max_steps")

        self.warmup_steps    = int(warmup_steps)
        self.max_steps       = int(max_steps)
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min         = float(eta_min)

        # PyTorch uses `last_epoch` internally; we repurpose it as "last_step".
        super().__init__(optimizer, last_epoch=last_step)

    # Core LR computation when `step()` is called without explicit step
    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last computed learning rate, call `get_last_lr()` instead.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        if 0 < self.last_epoch < self.warmup_steps:
            denom = max(self.warmup_steps - 1, 1)
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / denom
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        if self.last_epoch == self.warmup_steps:
            return list(self.base_lrs)

        # Cosine phase (by steps)
        T_cur   = self.last_epoch - self.warmup_steps
        T_total = max(self.max_steps - self.warmup_steps, 1)

        if (self.last_epoch - 1 - self.max_steps) % (2 * T_total) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / T_total)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_total))
            for base_lr in self.base_lrs
        ]

    # Closed-form LR when `step(step_idx)` is used
    def _get_closed_form_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_steps:
            if self.last_epoch == 0:
                return [self.warmup_start_lr] * len(self.base_lrs)
            denom = max(self.warmup_steps, 1)
            return [
                self.warmup_start_lr
                + self.last_epoch * (base_lr - self.warmup_start_lr) / denom
                for base_lr in self.base_lrs
            ]

        T_cur   = self.last_epoch - self.warmup_steps
        T_total = max(self.max_steps - self.warmup_steps, 1)

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_total))
            for base_lr in self.base_lrs
        ]
# ------------------------------------------------------------------------------#