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

#---------------------------- All Metrics Combined -----------------------------#
def all_metrics(pred_binary, pred_logits, gt):
    """
    Computes all metrics: DSC, IoU, SSIM, 
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
    }

# ---------------------------  End --------------------------------------------#
