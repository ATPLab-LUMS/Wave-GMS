# ------------------------------------------------------------------------------#
#
# File name                 : metrics.py
# Purpose                   : Segmentation metrics: Dice, IoU, HD95, SSIM (global/region/object/combined)
# Usage                     : from tools.metrics import all_metrics, dice_score, iou_score, hd95_score
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : August 26, 2025
# ------------------------------------------------------------------------------#

# --------------------------- Module Imports -----------------------------------#
from __future__                         import annotations

from typing                              import Tuple, Dict, Any
import numpy                             as np

from medpy                               import metric
from sklearn.metrics                     import confusion_matrix
# ------------------------------------------------------------------------------#

__all__ = [
    "paper_dice_iou",
    "dice_score",
    "iou_score",
    "hd95_score",
    "ssim",
    "ssim_region",
    "ssim_object",
    "ssim_combined",
    "all_metrics",
]


# --------------------------- Helpers / Conversions ----------------------------#
def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/list to np.ndarray without copying when possible."""
    if isinstance(x, np.ndarray):
        return x
    try:
        # torch.Tensor path (avoid importing torch globally)
        import torch  # lazy import
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _binarize(x: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Return uint8 mask in {0,1} with the same shape."""
    return (x >= thr).astype(np.uint8)


# --------------------------- Dice / IoU (paper-style) -------------------------#
def paper_dice_iou(y_pred, y_true) -> Tuple[float, float]:
    """
    Replicates the paper's Dice/IoU via confusion matrix on hard masks.
    """
    preds = _binarize(_to_numpy(y_pred).reshape(-1))
    gts   = _binarize(_to_numpy(y_true).reshape(-1))

    # Handle all-zero edge cases robustly
    if preds.size == 0:
        return 0.0, 0.0

    cm = confusion_matrix(gts, preds, labels=[0, 1])
    if cm.size != 4:
        # Handle degenerate cases where one class missing
        TN = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        TP = cm[-1, -1] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        FP = FN = 0
    else:
        TN, FP, FN, TP = cm.ravel()

    denom_dice = 2 * TP + FP + FN
    denom_iou  = TP + FP + FN

    dsc  = (2.0 * TP / denom_dice) if denom_dice > 0 else 0.0
    miou = (1.0 * TP / denom_iou)  if denom_iou  > 0 else 0.0
    return float(dsc), float(miou)


# --------------------------- Dice Score (DSC) ---------------------------------#
def dice_score(y_pred, y_true, eps: float = 1e-7) -> float:
    """
    Soft-to-hard Dice on binary masks in {0,1}.
    """
    yp = _binarize(_to_numpy(y_pred))
    yt = _binarize(_to_numpy(y_true))

    inter = float(np.sum(yp * yt))
    denom = float(np.sum(yp) + np.sum(yt))
    return (2.0 * inter) / (denom + eps)


# --------------------------- IoU Score ----------------------------------------#
def iou_score(y_pred, y_true, eps: float = 1e-7) -> float:
    """
    Intersection-over-Union on binary masks in {0,1}.
    """
    yp = _binarize(_to_numpy(y_pred))
    yt = _binarize(_to_numpy(y_true))

    inter = float(np.sum(yp * yt))
    union = float(np.sum(yp) + np.sum(yt) - inter)
    return inter / (union + eps)


# --------------------------- Hausdorff Distance (HD95) ------------------------#
def hd95_score(y_pred, y_true) -> float:
    """
    95th percentile Hausdorff distance between two 2D binary masks.

    Returns 0.0 if any mask is empty or inputs are not 2D.
    """
    yp = _binarize(_to_numpy(y_pred))
    yt = _binarize(_to_numpy(y_true))

    if yp.ndim != 2 or yt.ndim != 2:
        raise ValueError("hd95_score expects 2D masks (H, W).")
    if yp.sum() == 0 or yt.sum() == 0:
        # Common convention in medical seg eval: define HD95=0 for empty cases
        return 0.0

    return float(metric.binary.hd95(yp, yt))


# --------------------------- SSIM (global) ------------------------------------#
def ssim(pred, gt, eps: float = 1e-7) -> float:
    """
    Structural Similarity Index (simple global form).
    Inputs are flattened into vectors in [0,1].
    """
    p = _to_numpy(pred).astype(np.float64).reshape(-1)
    g = _to_numpy(gt).astype(np.float64).reshape(-1)

    mp, mg = p.mean(), g.mean()
    sp, sg = p.std(), g.std()
    cov    = np.mean((p - mp) * (g - mg))

    luminance = (2 * mp * mg) / (mp * mp + mg * mg + eps)
    contrast  = (2 * sp * sg) / (sp * sp + sg * sg + eps)
    structure = cov / (sp * sg + eps)

    return float(luminance * contrast * structure)


# --------------------------- Region-aware SSIM --------------------------------#
def ssim_region(pred, gt, eps: float = 1e-7) -> float:
    """
    Region-aware SSIM (quadrant split using GT centroid), per ICCV'17 metric.
    pred : 2D array in [0,1]
    gt   : 2D binary-like array (thresholded at 0.5)
    """
    P = _to_numpy(pred).astype(np.float64)
    G = _to_numpy(gt).astype(np.float64)
    if P.shape != G.shape:
        raise ValueError(f"Shapes must match: {P.shape} vs {G.shape}")
    if P.size == 0:
        return 0.0
    if P.min() < 0.0 or P.max() > 1.0:
        raise ValueError("pred should be in [0, 1]")

    Gb = (G > 0.5).astype(bool)

    def centroid(mask: np.ndarray) -> Tuple[int, int]:
        h, w = mask.shape
        if mask.sum() == 0:
            return round(w / 2), round(h / 2)
        total = mask.sum()
        xs    = np.arange(w)
        ys    = np.arange(h)
        X     = round(np.sum(mask.sum(axis=0) * xs) / total)
        Y     = round(np.sum(mask.sum(axis=1) * ys) / total)
        return X, Y

    def split(arr: np.ndarray, X: int, Y: int):
        h, w = arr.shape
        LT = arr[:Y, :X]
        RT = arr[:Y, X:w]
        LB = arr[Y:h, :X]
        RB = arr[Y:h, X:w]
        return LT, RT, LB, RB

    def weights(h: int, w: int, X: int, Y: int) -> Tuple[float, float, float, float]:
        area = float(h * w)
        w1   = (X * Y) / area
        w2   = ((w - X) * Y) / area
        w3   = (X * (h - Y)) / area
        w4   = 1.0 - w1 - w2 - w3
        return w1, w2, w3, w4

    def ssim_region_core(Preg: np.ndarray, Greg: np.ndarray) -> float:
        h, w = Preg.shape
        N    = float(h * w)
        if N <= 1:
            return 0.0
        x  = float(Preg.mean())
        y  = float(Greg.mean())
        vx = float(np.sum((Preg - x) ** 2) / (N - 1 + eps))
        vy = float(np.sum((Greg - y) ** 2) / (N - 1 + eps))
        c  = float(np.sum((Preg - x) * (Greg - y)) / (N - 1 + eps))
        alpha = 4.0 * x * y * c
        beta  = (x * x + y * y) * (vx + vy)
        if alpha != 0.0:
            return float(alpha / (beta + eps))
        return 1.0 if beta == 0.0 else 0.0

    X, Y                 = centroid(Gb)
    G1, G2, G3, G4       = split(Gb.astype(np.float64), X, Y)
    P1, P2, P3, P4       = split(P, X, Y)
    w1, w2, w3, w4       = weights(*Gb.shape, X, Y)

    Q1 = ssim_region_core(P1, G1)
    Q2 = ssim_region_core(P2, G2)
    Q3 = ssim_region_core(P3, G3)
    Q4 = ssim_region_core(P4, G4)

    return float(w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4)


# --------------------------- Object-aware SSIM --------------------------------#
def ssim_object(pred, gt, eps: float = 1e-7) -> float:
    """
    Object-aware SSIM: evaluates foreground and background separately, then mixes
    by GT foreground ratio.
    """
    P = _to_numpy(pred).astype(np.float64)
    G = _to_numpy(gt).astype(np.float64)
    if P.shape != G.shape:
        raise ValueError(f"Shapes must match: {P.shape} vs {G.shape}")
    if P.size == 0:
        return 0.0
    if P.min() < 0.0 or P.max() > 1.0:
        raise ValueError("pred should be in [0, 1]")

    Gb = (G > 0.5).astype(bool)

    def obj_score(pred_map: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        vals  = pred_map[mask]
        mean  = float(vals.mean())
        std   = float(vals.std())
        # identical to paper's smooth form
        return float((2.0 * mean) / (mean * mean + 1.0 + std + eps))

    O_fg = obj_score(P, Gb)
    O_bg = obj_score(1.0 - P, ~Gb)

    u = float(Gb.mean())
    return float(u * O_fg + (1.0 - u) * O_bg)


# --------------------------- Combined SSIM ------------------------------------#
def ssim_combined(pred, gt, alpha: float = 0.5, eps: float = 1e-7) -> float:
    """
    Combined SSIM: alpha * object-aware + (1-alpha) * region-aware.
    Handles special cases when GT is all-0 or all-1.
    """
    P = _to_numpy(pred).astype(np.float64)
    G = _to_numpy(gt).astype(np.float64)
    if P.shape != G.shape:
        raise ValueError(f"Shapes must match: {P.shape} vs {G.shape}")
    if P.size == 0:
        return 0.0
    if P.min() < 0.0 or P.max() > 1.0:
        raise ValueError("pred should be in [0, 1]")

    Gb = (G > 0.5).astype(bool)
    y  = float(Gb.mean())
    x  = float(P.mean())

    if y == 0.0:   # empty GT → prefer background
        return float(1.0 - x)
    if y == 1.0:   # full GT → prefer foreground
        return float(x)

    So = ssim_object(P, Gb, eps=eps)   # object-aware
    Sr = ssim_region(P, Gb, eps=eps)   # region-aware
    Q  = max(alpha * So + (1.0 - alpha) * Sr, 0.0)
    return float(Q)


# --------------------------- All metrics (dict) -------------------------------#
def all_metrics(pred_binary, pred_logits, gt, alpha: float = 0.5) -> Dict[str, float]:
    """
    Convenience wrapper computing the whole suite.

    Args:
        pred_binary : binary mask in {0,1} (any type compatible with _to_numpy)
        pred_logits : probability/logit-like map in [0,1] preferred (same HxW)
        gt          : binary GT mask in {0,1}
        alpha       : combined-SSIM weighting (object vs region)
    """
    pb = _to_numpy(pred_binary)
    pl = _to_numpy(pred_logits)
    gt = _to_numpy(gt)

    return {
        "DSC"          : float(dice_score(pb, gt)),
        "IoU"          : float(iou_score(pb, gt)),
        "HD95"         : float(hd95_score(pb, gt)),
        "SSIM"         : float(ssim(pl, gt)),
        "SSIM_region"  : float(ssim_region(pl, gt)),
        "SSIM_object"  : float(ssim_object(pl, gt)),
        "SSIM_combined": float(ssim_combined(pl, gt, alpha=alpha)),
    }
# ------------------------------------------------------------------------------#