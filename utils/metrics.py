# ------------------------------------------------------------------------------#
#
# File name   : metrics.py
# Purpose     : Implements segmentation evaluation metrics: Dice, IoU, SSIM,
#               region-aware/object-aware/combined SSIM for binary maps.
# Usage       : Imported by validation and analysis scripts.
#
# Authors     : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email       : 24100033@lums.edu.pk, 202410001@lums.edu.pk,
#               hassan.mohyuddin@lums.edu.pk
#
# Last Modified: June 23, 2025
# ---------------------------------- Module Imports --------------------------------------------#
import torch

import numpy as np

from medpy import metric

# --------------------------- Dice Score (DSC) ---------------------------------#
def dice_score(y_pred, y_true, eps = 1e-7):
    """
    Computes Dice Score (F1/DSC) for binary arrays.
    """

    # print(f'y_pred.shape: {y_pred.shape}, y_true.shape: {y_true.shape}')

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
    # y_pred = (y_pred >= 0.5).astype(np.uint8)
    # y_true = (y_true >= 0.5).astype(np.uint8)

    # if y_pred.sum() == 0 or y_true.sum() == 0:
    #     return np.nan

    # dt_true = distance_transform_edt(1 - y_true)
    # dt_pred = distance_transform_edt(1 - y_pred)

    # surface_pred = (y_pred - (distance_transform_edt(1 - y_pred) > 0)).astype(bool)
    # surface_true = (y_true - (distance_transform_edt(1 - y_true) > 0)).astype(bool)

    # dist_pred_to_true = dt_true[surface_pred]
    # dist_true_to_pred = dt_pred[surface_true]

    # all_dists = np.concatenate([dist_pred_to_true, dist_true_to_pred])
    # return np.percentile(all_dists, 95)

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
def ssim_region(pred, gt, mask = None):
    """
    Computes region-aware SSIM by splitting map into four quadrants at centroid.
    Args:
        pred, gt: 2D arrays, values in [0,1].
    Returns:
        Weighted sum of quadrant SSIMs.
    """
    if mask is None:
        mask = (gt > 0.5).astype(np.uint8)

    # Compute centroid of foreground in GT
    indices = np.argwhere(mask)
    if indices.size == 0:
        return 0.0

    centroid = indices.mean(axis=0).astype(int)
    h, w     = gt.shape
    h_c, w_c = centroid

    blocks = [
        ((slice(0, h_c), slice(0, w_c))),      # top-left
        ((slice(0, h_c), slice(w_c, w))),      # top-right
        ((slice(h_c, h), slice(0, w_c))),      # bottom-left
        ((slice(h_c, h), slice(w_c, w))),      # bottom-right
    ]

    # Weight = fraction of GT foreground pixels in block
    total_fg             = mask.sum()
    ssim_blocks, weights = [], []

    for b in blocks:
        gt_block   = gt[b]
        pred_block = pred[b]
        mask_block = mask[b]
        weight     = mask_block.sum() / (total_fg + 1e-8)
        if mask_block.sum() == 0:
            ssim_val = 0.0
        else:
            ssim_val = ssim(pred_block.flatten(), gt_block.flatten())
        ssim_blocks.append(ssim_val)
        weights.append(weight)

    return np.sum(np.array(ssim_blocks) * np.array(weights))

# --------------------------- Object-aware SSIM --------------------------------#
def ssim_object(pred, gt, lam = 0.5, eps = 1e-7):
    """
    Computes object-aware SSIM as in paper.
    Args:
        pred, gt: 2D arrays, values in [0,1].
        lam:     weighting for dispersion term.
    Returns:
        Object-aware similarity (So).
    """
    fg_mask = (gt > 0.5)
    bg_mask = ~fg_mask

    # Foreground
    x_fg = pred[fg_mask]
    y_fg = gt[fg_mask]

    mean_x_fg = x_fg.mean() if x_fg.size > 0 else 0.0
    mean_y_fg = y_fg.mean() if y_fg.size > 0 else 0.0
    std_x_fg  = x_fg.std()  if x_fg.size > 0 else 0.0

    # Object level dissimilarity and similarity
    D_fg = ((mean_x_fg ** 2 + mean_y_fg ** 2) / (2 * mean_x_fg * mean_y_fg + eps)) + lam * (std_x_fg / (mean_x_fg + eps))
    O_fg = 1. / (D_fg + eps)

    # Background
    x_bg = pred[bg_mask]
    y_bg = gt[bg_mask]

    mean_x_bg = x_bg.mean() if x_bg.size > 0 else 0.0
    mean_y_bg = y_bg.mean() if y_bg.size > 0 else 0.0
    std_x_bg  = x_bg.std()  if x_bg.size > 0 else 0.0

    D_bg = ((mean_x_bg ** 2 + mean_y_bg ** 2) / (2 * mean_x_bg * mean_y_bg + eps)) + lam * (std_x_bg / (mean_x_bg + eps))
    O_bg = 1. / (D_bg + eps)

    # µ: ratio of GT foreground area
    mu = fg_mask.sum() / (gt.size + eps)

    return mu * O_fg + (1 - mu) * O_bg

# --------------------------- Combined SSIM (final metric) ---------------------#
def ssim_combined(pred, gt, alpha = 0.5, lam = 0.5):
    """
    Final measure: weighted sum of object-aware and region-aware SSIM.
    """
    pred = np.asarray(pred, dtype = np.float64)
    gt   = np.asarray(gt, dtype   = np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes must match: {pred.shape}, {gt.shape}")

    # Normalize
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt   = (gt   - gt.min())   / (gt.max()   - gt.min()   + 1e-8)

    So = ssim_object(pred, gt, lam=lam)
    Sr = ssim_region(pred, gt)

    return alpha * So + (1 - alpha) * Sr

#---------------------------- All Metrics Combined -----------------------------#
def all_metrics(pred_binary, pred_logits, gt, alpha = 0.5, lam = 0.5):
    """
    Computes all metrics: DSC, IoU, SSIM, region-aware SSIM, object-aware SSIM,
    and combined SSIM.
    Args:
        pred_binary: 2D binary array, predicted segmentation.
        pred_logits: 2D logits array, predicted segmentation
        gt:          2D binary array, ground truth segmentation.
        alpha:   weight for object-aware vs region-aware SSIM.
        lam:     weight for dispersion term in object-aware SSIM.
    Returns:
        dict of all metrics.
    """
    # Note pass pred_binary for DSC and IOU and for the rest pred_logits
    return {
        'DSC'              : dice_score(pred_binary, gt).item(),
        'IoU'              : iou_score(pred_binary, gt).item(),
        'HD95'             : hd95_score(pred_binary, gt),
        'SSIM'             : ssim(pred_logits.flatten(), gt.flatten()).item(),
        'SSIM_region'      : ssim_region(pred_logits, gt).item(),
        'SSIM_object'      : ssim_object(pred_logits, gt, lam = lam).item(),
        'SSIM_combined'    : ssim_combined(pred_logits, gt, alpha = alpha, lam = lam).item()
    }

# ---------------------------  End --------------------------------------------#
