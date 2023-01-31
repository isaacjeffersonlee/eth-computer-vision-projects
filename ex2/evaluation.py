import numpy as np


def calculate_iou(gt_mask, pred_mask):
    gt_mask_bin = (gt_mask == 255).astype(np.uint8)
    pred_mask_bin = (pred_mask == 255).astype(np.uint8)

    intersection = (gt_mask_bin * pred_mask_bin).sum()
    union = gt_mask_bin.sum() + pred_mask_bin.sum() - intersection
    return intersection / union
