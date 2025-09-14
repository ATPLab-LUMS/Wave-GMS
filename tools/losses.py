# ------------------------------------------------------------------------------#
#
# File name                 : losses.py
# Purpose                   : Segmentation losses (Weighted BCE, Dice, Combo)
# Usage                     : from tools.losses import WeightedBCE, WeightedDiceLoss, WeightedDiceBCE
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
import torch.nn.functional        as F
# ------------------------------------------------------------------------------#

__all__ = ["WeightedBCE", "WeightedDiceLoss", "WeightedDiceBCE"]


# ------------------------------- Weighted BCE ---------------------------------#
class WeightedBCE(nn.Module):
    """
    Class-weighted BCE with optional logits input.

    Args:
        weights        : [w_pos, w_neg] or tensor of shape (2,)
        from_logits    : if True, applies BCEWithLogits; else expects probabilities in [0,1]
        eps            : numerical stability when normalizing positive/negative counts
    """
    def __init__(self, weights=(0.5, 0.5), from_logits: bool = False, eps: float = 1e-12):
        super().__init__()
        self.w_pos, self.w_neg = float(weights[0]), float(weights[1])
        self.from_logits       = from_logits
        self.eps               = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten per-sample then average over batch
        B = pred.shape[0]
        p = pred.flatten(1)
        t = target.flatten(1)
        assert p.shape == t.shape, "Prediction/Target shapes must match after flatten."

        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        else:
            bce = F.binary_cross_entropy(p, t, reduction="none")

        # Class masks
        pos = (t > 0.5).float()
        neg = 1.0 - pos

        # Normalize by counts per batch to avoid imbalance scaling with image size
        pos_count = pos.sum(dim=1, keepdim=True) + self.eps
        neg_count = neg.sum(dim=1, keepdim=True) + self.eps

        loss = (
            self.w_pos * (pos * bce) / pos_count +
            self.w_neg * (neg * bce) / neg_count
        ).sum(dim=1).mean()  # sum over pixels, mean over batch

        return loss


# ----------------------------- Weighted Dice Loss -----------------------------#
class WeightedDiceLoss(nn.Module):
    """
    Weighted soft Dice loss. If `from_logits=True`, a sigmoid is applied.

    Args:
        weights        : [w_bg, w_fg] applied via target-based weighting
        from_logits    : apply sigmoid to predictions if True
        smooth         : numerical stability
    """
    def __init__(self, weights=(0.5, 0.5), from_logits: bool = False, smooth: float = 1e-5):
        super().__init__()
        self.w_bg, self.w_fg = float(weights[0]), float(weights[1])
        self.from_logits     = from_logits
        self.smooth          = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]
        p = pred.flatten(1)
        t = target.flatten(1)
        assert p.shape == t.shape, "Prediction/Target shapes must match after flatten."

        if self.from_logits:
            p = torch.sigmoid(p)

        # target-driven per-pixel weights in [w_bg, w_fg]
        w = t.detach() * (self.w_fg - self.w_bg) + self.w_bg
        p = w * p
        t = w * t

        inter = (p * t).sum(dim=1)
        union = (p * p).sum(dim=1) + (t * t).sum(dim=1)
        dice  = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)

        return dice.mean()


# ---------------------------- Weighted Dice + BCE -----------------------------#
class WeightedDiceBCE(nn.Module):
    """
    Combo loss = alpha * Dice + beta * BCE.

    Args:
        dice_weight    : α
        BCE_weight     : β
        from_logits    : pass through to both components
        weights_dice   : (w_bg, w_fg) for Dice
        weights_bce    : (w_pos, w_neg) for BCE
    """
    def __init__(
        self,
        dice_weight: float = 1.0,
        BCE_weight : float = 1.0,
        from_logits: bool  = False,
        weights_dice=(0.5, 0.5),
        weights_bce =(0.5, 0.5),
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.BCE_weight  = float(BCE_weight)

        self.dice = WeightedDiceLoss(weights=weights_dice, from_logits=from_logits)
        self.bce  = WeightedBCE     (weights=weights_bce,  from_logits=from_logits)

    @torch.no_grad()
    def _hard_dice(self, pred: torch.Tensor, target: torch.Tensor, from_logits: bool = False) -> torch.Tensor:
        """Utility for logging: thresholded dice on probabilities."""
        p = torch.sigmoid(pred) if from_logits else pred
        p = (p >= 0.5).float()
        t = (target > 0.0).float()
        # reuse dice core (no weights, no grad)
        inter = (p * t).flatten(1).sum(dim=1)
        union = p.flatten(1).sum(dim=1) + t.flatten(1).sum(dim=1)
        dice  = (2.0 * inter + 1e-5) / (union + 1e-5)
        return dice.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = self.dice(pred, target)
        b = self.bce (pred, target)
        return self.dice_weight * d + self.BCE_weight * b