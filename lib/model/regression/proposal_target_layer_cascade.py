from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from core.config import cfg
from .bbox_transform import bbox_transform
import pdb

def _get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form b x N x (class, tx, ty, tw, th)
    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).
    Returns:
        bbox_target (ndarray): b x N x 4K blob of regression targets
        bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
    """
    BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    rois_per_image = labels_batch.size(0)
    clss = labels_batch
    bbox_targets = bbox_target_data.new(rois_per_image, 4).zero_()
    bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

    inds = torch.nonzero(clss > 0).view(-1)
    for i in range(inds.numel()):
        ind = inds[i]
        bbox_targets[ind, :] = bbox_target_data[ind, :]
        bbox_inside_weights[ind, :] = BBOX_INSIDE_WEIGHTS

    return bbox_targets, bbox_inside_weights


def _compute_targets_pytorch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)

    targets = bbox_transform(ex_rois, gt_rois)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - BBOX_NORMALIZE_MEANS.expand_as(targets))
                    / BBOX_NORMALIZE_STDS.expand_as(targets))

    return targets
