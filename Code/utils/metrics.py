# ------------------------------------------------------------------------------#
#
# File name   : metrics.py
# Purpose     : Implements segmentation evaluation metrics: Dice, IoU, SSIM,
#               region-aware/object-aware/combined SSIM for binary maps.
# Usage       : Imported by validation and analysis scripts.
#
# Authors     : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email       : 24100033@lums.edu.pk, 24020001@lums.edu.pk,
#               hassan.mohyuddin@lums.edu.pk
#
# Last Modified: June 23, 2025

# Note: Implementation is adapted from the MATLAB code released with
#       “Structure‑measure: A New Way to Evaluate Foreground Maps”
#       (Deng‑Ping Fan et al., ICCV 2017).
# ---------------------------------- Module Imports --------------------------------------------#
import torch
import numpy as np
from medpy import metric

"""
Papers implementation of dice and iou. both give same results

    preds = np.array(y_pred).reshape(-1)
    gts   = np.array(y_true).reshape(-1)

    y_pre = np.where(preds > 0.5, 1, 0)
    y_gt  = np.where(gts > 0.5, 1, 0)

    confusion = confusion_matrix(y_gt, y_pre)
    TN, FP, FN, TP = confusion.ravel()

    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou      = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    return f1_or_dsc, miou

"""

# --------------------------- Dice Score (DSC) ---------------------------------#
def dice_score(y_pred, y_true, eps = 1e-7):
    """
    Computes Dice Score (F1/DSC) for binary arrays.
    """

    y_pred = (y_pred >= 0.5).astype(np.uint8)
    y_true = (y_true >= 0.5).astype(np.uint8)

    intersection = np.sum(y_pred * y_true)
    denominator  = np.sum(y_pred) + np.sum(y_true)

    return 2. * intersection / (denominator + eps)

# --------------------------- IoU Score ----------------------------------------#
def iou_score(y_pred, y_true, eps = 1e-7):
    """
    Computes Intersection-over-Union (IoU) for binary arrays.
    """
    y_pred = (y_pred >= 0.5).astype(np.uint8)
    y_true = (y_true >= 0.5).astype(np.uint8)

    intersection = np.sum(y_pred * y_true)
    union        = np.sum(y_pred) + np.sum(y_true) - intersection

    return intersection / (union + eps)

# --------------------------- Hausdorff Distance (HD95) ------------------------#
def hd95_score(y_pred, y_true):
    """
    Computes 95th percentile Hausdorff Distance (HD95) between two binary masks.
    """

    # Check if y_pred and y_true are binary masks
    if y_pred.ndim != 2 or y_true.ndim != 2:
        raise ValueError("y_pred and y_true must be 2D binary masks.")
    if y_pred.sum() == 0 or y_true.sum() == 0:
        print("One of the masks is empty, returning 0.0 for HD95.")
        return 0.0

    hd95 = metric.binary.hd95(y_pred, y_true)
    return hd95

# --------------------------- SSIM ---------------------------------------------#
def ssim(pred, gt, eps = 1e-7):
    """
    Computes Structural Similarity Index (SSIM) between two images.
    Args:
        pred, gt: flattened arrays, values in [0,1].
    Returns:
        SSIM value.
    """
    pred = pred.astype(np.float64)
    gt   = gt.astype(np.float64)

    mean_pred = pred.mean()
    mean_gt   = gt.mean()
    std_pred  = pred.std()
    std_gt    = gt.std()
    cov       = np.mean((pred - mean_pred) * (gt - mean_gt))

    # 3 components: luminance, contrast, structure
    luminance = (2 * mean_pred * mean_gt) / (mean_pred**2 + mean_gt**2 + eps)
    contrast  = (2 * std_pred * std_gt)   / (std_pred**2 + std_gt**2 + eps)
    structure = cov / (std_pred * std_gt + eps)

    return luminance * contrast * structure

# --------------------------- Region-aware SSIM (quadrant split) ---------------#
def ssim_region(pred, gt, eps=1e-7):
    """
    Args:
        pred: 2D array, values in [0,1] (type: float64).
        gt: 2D binary array (type: bool or uint8, thresholded at 0.5 if not binary).
        eps: Small constant for numerical stability (default: 1e-7).
    Returns:
        Region-aware similarity score (Q).
    """
    # Input validation
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes must match: {pred.shape}, {gt.shape}")
    if pred.size == 0:
        return 0.0
    if pred.max() > 1.0 or pred.min() < 0.0:
        raise ValueError("pred should be in the range [0, 1]")

    # Convert gt to binary (threshold at 0.5 if not binary)
    gt = (gt > 0.5).astype(bool)

    def centroid(GT):
        """Compute the centroid of the ground truth."""
        rows, cols = GT.shape
        if GT.sum() == 0:
            X = round(cols / 2)
            Y = round(rows / 2)
        else:
            total = GT.sum()
            i = np.arange(cols)
            j = np.arange(rows)
            X = round(np.sum(GT.sum(axis=0) * i) / total)
            Y = round(np.sum(GT.sum(axis=1) * j) / total)
        return X, Y

    def divideGT(GT, X, Y):
        """Divide GT into 4 regions and compute weights."""
        hei, wid = GT.shape
        area = wid * hei
        # Copy the 4 regions
        LT = GT[:Y, :X]
        RT = GT[:Y, X: wid]
        LB = GT[Y: hei, :X]
        RB = GT[Y: hei, X: wid]
        # Compute weights proportional to GT foreground area
        w1 = (X * Y) / area
        w2 = ((wid - X) * Y) / area
        w3 = (X * (hei - Y)) / area
        w4 = 1.0 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def divide_prediction(prediction, X, Y):
        """Divide prediction into 4 regions based on GT centroid."""
        hei, wid = prediction.shape
        LT = prediction[:Y, :X]
        RT = prediction[:Y, X:wid]
        LB = prediction[Y:hei, :X]
        RB = prediction[Y:hei, X:wid]
        return LT, RT, LB, RB

    def ssim_calculate(prediction, GT):
        """Compute SSIM for a region."""
        dGT = GT.astype(np.float64)
        hei, wid = prediction.shape
        N   = wid * hei
        # Compute means
        x = np.mean(prediction)
        y = np.mean(dGT)
        # Compute variances
        sigma_x2 = np.sum((prediction - x)**2) / (N - 1 + eps)
        sigma_y2 = np.sum((dGT - y)**2) / (N - 1 + eps)
        # Compute covariance
        sigma_xy = np.sum((prediction - x) * (dGT - y)) / (N - 1 + eps)
        # Compute SSIM
        alpha = 4 * x * y * sigma_xy
        beta  = (x**2 + y**2) * (sigma_x2 + sigma_y2)
        if alpha != 0:
            Q = alpha / (beta + eps)
        elif alpha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0.0
        return Q

    # Compute centroid
    X, Y = centroid(gt)
    # Divide GT and prediction into 4 regions
    GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    pred_1, pred_2, pred_3, pred_4         = divide_prediction(pred, X, Y)

    # Compute SSIM for each region
    Q1 = ssim_calculate(pred_1, GT_1)
    Q2 = ssim_calculate(pred_2, GT_2)
    Q3 = ssim_calculate(pred_3, GT_3)
    Q4 = ssim_calculate(pred_4, GT_4)
    # Combine scores with weights
    Q  = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
    return Q

# --------------------------- Object-aware SSIM --------------------------------#
def ssim_object(pred, gt, eps=1e-7):
    """
    Args:
        pred: 2D array, values in [0,1] (type: float64).
        gt: 2D binary array (type: bool or uint8, thresholded at 0.5 if not binary).
        eps: Small constant for numerical stability (default: 1e-7).
    Returns:
        Object-aware similarity score (Q).
    """
    # Input validation
    pred = np.asarray(pred, dtype = np.float64)
    gt   = np.asarray(gt, dtype = np.float64)

    if pred.shape != gt.shape:
        raise ValueError(f"Shapes must match: {pred.shape}, {gt.shape}")
    if pred.size == 0:
        return 0.0
    if pred.max() > 1.0 or pred.min() < 0.0:
        raise ValueError("pred should be in the range [0, 1]")

    # Convert gt to binary (threshold at 0.5 if not binary)
    gt = (gt > 0.5).astype(bool)

    def object_score(prediction, mask):
        """Helper function to compute object similarity for a region."""
        if not np.any(mask):
            return 0.0

        # Compute mean and std of prediction in the masked region
        x       = np.mean(prediction[mask])
        sigma_x = np.std(prediction[mask])

        # Compute object similarity: 2 * mean / (mean^2 + 1 + std + eps)
        score = (2.0 * x) / (x**2 + 1.0 + sigma_x + eps)
        return score

    # Foreground: mask out background
    pred_fg = pred.copy()
    pred_fg[~gt] = 0
    O_fg    = object_score(pred_fg, gt)

    # Background: invert prediction and mask out foreground
    pred_bg     = 1.0 - pred
    pred_bg[gt] = 0
    O_bg        = object_score(pred_bg, ~gt)

    # Combine foreground and background scores
    u = np.mean(gt)  # Fraction of foreground pixels
    return u * O_fg + (1 - u) * O_bg

# --------------------------- Combined SSIM (final metric) ---------------------#
def ssim_combined(pred, gt, alpha=0.5, eps=1e-7):
    """
    Args:
        pred: 2D array, values in [0,1] (type: float64).
        gt: 2D binary array (type: bool or uint8, thresholded at 0.5 if not binary).
        alpha: Weight for object-aware vs region-aware SSIM (default: 0.5).
        eps: Small constant for numerical stability (default: 1e-7).
    Returns:
        Combined similarity score (Q).
    """
    # Input validation
    pred = np.asarray(pred, dtype=np.float64)
    gt   = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes must match: {pred.shape}, {gt.shape}")
    if pred.size == 0:
        return 0.0
    if pred.max() > 1.0 or pred.min() < 0.0:
        raise ValueError("pred should be in the range [0, 1]")

    # Convert gt to binary (threshold at 0.5 if not binary)
    gt = (gt > 0.5).astype(bool)

    # Compute mean of ground truth
    y = np.mean(gt)

    # Handle special cases
    if y == 0:  # GT is completely black
        x = np.mean(pred)
        Q = 1.0 - x  # Only calculate intersection area
    elif y == 1:  # GT is completely white
        x = np.mean(pred)
        Q = x  # Only calculate intersection area
    else:
        # Combine object-aware and region-aware SSIM
        So = ssim_object(pred, gt, eps=eps)
        Sr = ssim_region(pred, gt, eps=eps)
        Q = alpha * So + (1 - alpha) * Sr
        Q = max(Q, 0.0)  # Ensure non-negative score

    return Q

#---------------------------- All Metrics Combined -----------------------------#
def all_metrics(pred_binary, pred_logits, gt, alpha=0.5):
    """
    Computes all metrics: DSC, IoU, SSIM, region-aware SSIM, object-aware SSIM,
    and combined SSIM.
    Args:
        pred_binary: 2D binary array, predicted segmentation.
        pred_logits: 2D logits array, predicted segmentation.
        gt: 2D binary array, ground truth segmentation.
        alpha: Weight for object-aware vs region-aware SSIM.
    Returns:
        dict of all metrics.
    """
    return {
        'DSC'           : dice_score(pred_binary, gt).item(),
        'IoU'           : iou_score(pred_binary, gt).item(),
        'HD95'          : hd95_score(pred_binary, gt),
        'SSIM'          : ssim(pred_logits.flatten(), gt.flatten()).item(),
        'SSIM_region'   : ssim_region(pred_logits, gt).item(),
        'SSIM_object'   : ssim_object(pred_logits, gt).item(),
        'SSIM_combined' : ssim_combined(pred_logits, gt, alpha=alpha).item()
    }

# ---------------------------  End --------------------------------------------#