# ------------------------------------------------------------------------------#
#
# File name                 : lr_scheduler.py
# Purpose                   : Custom learning rate scheduler with linear warmup
#                             followed by cosine annealing.
# Usage                     : Plug into PyTorch optimizer during training.
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : June 21, 2025
# ------------------------------------------------------------------------------#


# --------------------------- Module imports -----------------------------------#
import math, warnings

from typing                               import List
from torch.optim                          import Optimizer
from torch.optim.lr_scheduler             import _LRScheduler

# --------------------------- Custom LR Scheduler ------------------------------#
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Scheduler that linearly increases the learning rate for a given number of
    warmup epochs, then follows a cosine annealing schedule.

    Args:
        optimizer         : Wrapped PyTorch optimizer.
        warmup_epochs     : Number of warmup steps.
        max_epochs        : Total number of training epochs.
        warmup_start_lr   : Learning rate at the start of warmup.
        eta_min           : Minimum learning rate at the end of cosine annealing.
        last_epoch        : Last epoch index for resuming (default: -1).
    """

    def __init__(
        self,
        optimizer        : Optimizer,
        warmup_epochs    : int,
        max_epochs       : int,
        warmup_start_lr  : float = 0.0,
        eta_min          : float = 0.0,
        last_epoch       : int = -1,
    ) -> None:

        self.warmup_epochs     = warmup_epochs
        self.max_epochs        = max_epochs
        self.warmup_start_lr   = warmup_start_lr
        self.eta_min           = eta_min

        super().__init__(optimizer, last_epoch)


    # --------------------------- Get current LR -------------------------------#
    def get_lr(self) -> List[float]:
        """
        Compute the current learning rate based on the epoch (self.last_epoch).

        Returns:
            List of learning rates, one for each param group.
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        # Initial step
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)

        # Linear warmup phase
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        # Start of cosine phase
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        # Restart at cosine valley
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        # Standard cosine annealing
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) /
                          (self.max_epochs - self.warmup_epochs))) /
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) /
                          (self.max_epochs - self.warmup_epochs))) *
            (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]


    # --------------------------- Closed form LR --------------------------------#
    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed directly to `scheduler.step(epoch)`.

        Returns:
            List of closed-form learning rates.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch *
                (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) /
                          (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]

# --------------------------------- End -----------------------------------------#