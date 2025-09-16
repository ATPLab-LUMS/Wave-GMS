# ------------------------------------------------------------------------------#
#
# File name                 : metrics.py
# Purpose                   : Implements evaluation metrics for segmentation.
# Usage                     : from utils.metrics import dice_score, iou_score, all_metrics
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh,
#                             Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#                             hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : Sep 17, 2025
# Note                      : SSIM related metrics adapted from
#                             [https://arxiv.org/pdf/1708.00786.pdf]
# ------------------------------------------------------------------------------#

# ---------------------------------- Module Imports ----------------------------#
import torch
import numpy as np

from medpy import metric
# ------------------------------------------------------------------------------#


# --------------------------- Dice Score (DSC) ---------------------------------#
def dice_score(y_pred, y_true, eps: float = 1e-7) -> float:
    """Computes Dice Score (F1/DSC) for binary arrays."""
    y_pred = (y_pred >= 0.5).astype(np.uint8)
    y_true = (y_true >= 0.5).astype(np.uint8)

    intersection = np.sum(y_pred * y_true)
    denominator  = np.sum(y_pred) + np.sum(y_true)
    return 2.0 * intersection / (denominator + eps)


# --------------------------- IoU Score ----------------------------------------#
def iou_score(y_pred, y_true, eps: float = 1e-7) -> float:
    """Computes Intersection-over-Union (IoU) for binary arrays."""
    y_pred = (y_pred >= 0.5).astype(np.uint8)
    y_true = (y_true >= 0.5).astype(np.uint8)

    intersection = np.sum(y_pred * y_true)
    union        = np.sum(y_pred) + np.sum(y_true) - intersection
    return intersection / (union + eps)


# --------------------------- Hausdorff Distance (HD95) ------------------------#
def hd95_score(y_pred, y_true) -> float:
    """Computes 95th percentile Hausdorff Distance (HD95) between two binary masks."""
    if y_pred.ndim != 2 or y_true.ndim != 2:
        raise ValueError("y_pred and y_true must be 2D binary masks.")
    if y_pred.sum() == 0 or y_true.sum() == 0:
        print("One of the masks is empty, returning 0.0 for HD95.")
        return 0.0

    return metric.binary.hd95(y_pred, y_true)


# --------------------------- SSIM ---------------------------------------------#
def ssim(pred, gt, eps: float = 1e-7) -> float:
    """Computes Structural Similarity Index (SSIM) between two images."""
    pred = pred.astype(np.float64)
    gt   = gt.astype(np.float64)

    mean_pred, mean_gt = pred.mean(), gt.mean()
    std_pred, std_gt   = pred.std(), gt.std()
    cov                = np.mean((pred - mean_pred) * (gt - mean_gt))

    luminance = (2 * mean_pred * mean_gt) / (mean_pred**2 + mean_gt**2 + eps)
    contrast  = (2 * std_pred * std_gt)   / (std_pred**2 + std_gt**2 + eps)
    structure = cov / (std_pred * std_gt + eps)

    return luminance * contrast * structure


# --------------------------- Region-aware SSIM --------------------------------#
def ssim_region(pred, gt, eps: float = 1e-7) -> float:
    """Region-aware SSIM: splits GT into quadrants around its centroid."""
    # Validation
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes must match: {pred.shape}, {gt.shape}")
    if pred.size == 0:
        return 0.0
    if pred.max() > 1.0 or pred.min() < 0.0:
        raise ValueError("pred should be in [0, 1]")

    gt = (gt > 0.5).astype(bool)

    def centroid(GT):
        rows, cols = GT.shape
        if GT.sum() == 0:
            return round(cols / 2), round(rows / 2)
        total = GT.sum()
        X = round(np.sum(GT.sum(axis=0) * np.arange(cols)) / total)
        Y = round(np.sum(GT.sum(axis=1) * np.arange(rows)) / total)
        return X, Y

    def divide(GT, pred, X, Y):
        return (GT[:Y, :X], GT[:Y, X:], GT[Y:, :X], GT[Y:, X:],
                pred[:Y, :X], pred[:Y, X:], pred[Y:, :X], pred[Y:, X:])

    def ssim_calc(pred_r, GT_r):
        N = pred_r.size
        if N == 0: return 0.0
        x, y = pred_r.mean(), GT_r.mean()
        sigma_x2 = np.var(pred_r, ddof=1)
        sigma_y2 = np.var(GT_r, ddof=1)
        sigma_xy = np.mean((pred_r - x) * (GT_r - y))
        alpha = 4 * x * y * sigma_xy
        beta  = (x**2 + y**2) * (sigma_x2 + sigma_y2)
        if alpha != 0: return alpha / (beta + eps)
        return 1.0 if beta == 0 else 0.0

    X, Y = centroid(gt)
    GTs, preds = divide(gt, pred, X, Y)[:4], divide(gt, pred, X, Y)[4:]
    weights = [(X * Y), ((gt.shape[1]-X) * Y), (X * (gt.shape[0]-Y)), ((gt.shape[1]-X) * (gt.shape[0]-Y))]
    weights = [w / (gt.size) for w in weights]

    return sum(w * ssim_calc(p, g) for p, g, w in zip(preds, GTs, weights))


# --------------------------- Object-aware SSIM --------------------------------#
def ssim_object(pred, gt, eps: float = 1e-7) -> float:
    """Object-aware SSIM: foreground vs background similarity."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes must match: {pred.shape}, {gt.shape}")

    gt = (gt > 0.5).astype(bool)

    def obj_score(pred_r, mask):
        if not np.any(mask): return 0.0
        x, sigma_x = pred_r[mask].mean(), pred_r[mask].std()
        return (2 * x) / (x**2 + 1.0 + sigma_x + eps)

    O_fg = obj_score(pred, gt)
    O_bg = obj_score(1 - pred, ~gt)
    return np.mean(gt) * O_fg + (1 - np.mean(gt)) * O_bg


# --------------------------- Combined SSIM ------------------------------------#
def ssim_combined(pred, gt, alpha: float = 0.5, eps: float = 1e-7) -> float:
    """Weighted combination of object-aware and region-aware SSIM."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    gt = (gt > 0.5).astype(bool)

    y = gt.mean()
    if y == 0:  # GT all background
        return 1.0 - pred.mean()
    elif y == 1:  # GT all foreground
        return pred.mean()

    return max(alpha * ssim_object(pred, gt, eps) + (1 - alpha) * ssim_region(pred, gt, eps), 0.0)


# --------------------------- Aggregate All Metrics ----------------------------#
def all_metrics(pred_binary, pred_logits, gt, alpha: float = 0.5) -> dict:
    """Compute DSC, IoU, HD95, SSIM variants, and combined score."""
    return {
        "DSC"          : dice_score(pred_binary, gt).item(),
        "IoU"          : iou_score(pred_binary, gt).item(),
        "HD95"         : hd95_score(pred_binary, gt),
        "SSIM"         : ssim(pred_logits.flatten(), gt.flatten()).item(),
        "SSIM_region"  : ssim_region(pred_logits, gt).item(),
        "SSIM_object"  : ssim_object(pred_logits, gt).item(),
        "SSIM_combined": ssim_combined(pred_logits, gt, alpha).item(),
    }
# ------------------------------------------------------------------------------#