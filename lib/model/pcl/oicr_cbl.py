from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils.boxes as box_utils
from core.config import cfg
from model.regression.proposal_target_layer_cascade import _get_bbox_regression_labels_pytorch, _compute_targets_pytorch

import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax
import time
import os
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def OICR(boxes, cls_prob, im_labels, cls_prob_new, pred_boxes=None, info_dict=None, multigt=False, reg=False, vis_needed=None):
    if not isinstance(cls_prob, list):
        cls_prob = cls_prob.data.cpu().numpy()
        cls_prob_new = cls_prob_new.data.cpu().numpy()
        cls_prob_ori = cls_prob.copy()
        if cls_prob.shape[1] != im_labels.shape[1]:
            cls_prob = cls_prob[:, 1:]
        eps = 1e-9
        cls_prob[cls_prob < eps] = eps
        cls_prob[cls_prob > 1 - eps] = 1 - eps
    else:
        cls_prob_list = []
        cls_prob_ori = cls_prob.copy()
        for prob in cls_prob:
            prob = prob.data.cpu().numpy()
            if prob.shape[1] != im_labels.shape[1]:
                prob = prob[:, 1:]
            eps = 1e-9
            prob[prob < eps] = eps
            prob[prob > 1 - eps] = 1 - eps
            cls_prob_list.append(prob)
        cls_prob = cls_prob_list

    if multigt == True:
        proposals = _get_top_scoring_proposals_mtgt(boxes, cls_prob, im_labels, info_dict)
    else:
        proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels, info_dict)

    if cfg.OICR.Need_Reg:
        info_dict["cls_prob"] = cls_prob_ori
        info_dict["im_labels"] = im_labels
        info_dict["reg"] = reg
        labels, cls_loss_weights, labels_ori, bbox_targets, bbox_inside_weights, inds_info = \
            _sample_rois(boxes, proposals, 21, pred_boxes, info_dict)
        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.tensor(cls_loss_weights).cuda().detach(),
                'rois_labels' : labels_ori.long().cuda().detach(),
                'bbox_targets' : bbox_targets.cuda().detach(),
                'bbox_inside_weights' : bbox_inside_weights.cuda().detach(),
                'proposals': proposals,
                "inds_info": inds_info
                }

    else:
        labels, cls_loss_weights = _sample_rois(boxes, proposals, 21)

        return {'labels' : torch.FloatTensor(labels).cuda().detach(),
                'cls_loss_weights' : torch.tensor(cls_loss_weights).cuda().detach(),
                'proposals': proposals
                }


def _get_highest_score_proposals(boxes, cls_prob, im_labels, info_dict=None):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    gt_indices = []

    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:, i].copy()
            max_index = np.argmax(cls_prob_tmp)
            boxes_tmp = boxes[max_index, :].copy()
            gt_boxes = np.vstack((gt_boxes, boxes_tmp))
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
            gt_scores = np.vstack((gt_scores,
                cls_prob_tmp[max_index].reshape(-1, 1) ))  # * np.ones((1, 1), dtype=np.float32)))
            gt_indices.append(max_index)
            cls_prob[max_index, :] = 0 #in-place operation <- OICR code but I do not agree
    
    proposals = {'gt_boxes': gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores,
                 'gt_indices': gt_indices,
                 }

    return proposals



def _get_top_scoring_proposals_mtgt(boxes, cls_prob_list, im_labels, vis_needed):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :].copy()
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    gt_indices = []

    if cfg.OICR.CBL.RCNN_UseMTlayer_BRANCH == "last2-ens_rev":
        cls_prob_ens_0 = (cls_prob_list[-1] + cls_prob_list[-2]) / 2
        cls_prob_list = [cls_prob_ens_0] + [c for c in cls_prob_list[:-2]]
    elif cfg.OICR.CBL.RCNN_UseMTlayer_BRANCH == "last2-ens_last2":
        cls_prob_ens_0 = (cls_prob_list[-1] + cls_prob_list[-2]) / 2
        cls_prob_list = [cls_prob_ens_0, cls_prob_list[-1], cls_prob_list[-2]]
    

    tau = cfg.OICR.CBL.RCNN_UseMTlayer_IOUTHS

    select_ratio = cfg.OICR.CBL.TOPK_SELECT_RATIO
    score_ths_ratio = cfg.OICR.CBL.TOPK_SCORE_THS_RATIO
    select_num = int(len(boxes) * select_ratio)


    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            nms_dets_j_all = [ ]
            box_clusters = [ ]
            for j, cls_prob in enumerate(cls_prob_list):
                cls_prob_tmp = cls_prob[:, i].copy()
                ### sel top k
                score_ths = np.max(cls_prob_tmp) * score_ths_ratio
                select_num_high = len(np.where(cls_prob_tmp > score_ths)[0])
                select_num_cls = min(select_num, select_num_high)
                selected_inds = np.argsort(cls_prob_tmp)[-select_num_cls:]
                ### nms
                dets_j = np.hstack((boxes, cls_prob_tmp[:, np.newaxis])).astype(np.float32, copy=False)
                dets_j = dets_j[selected_inds]
                dets_j = np.vstack(dets_j)
                if cfg.OICR.CBL.RCNN_UseMTlayer_FILTER_NONMS:
                    used_branch = cfg.OICR.CBL.RCNN_UseMTlayer_SELECTBRANCH
                    if j == used_branch:
                        keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                        nms_dets = dets_j[keep, :]
                    else:
                        nms_dets = dets_j
                else:
                    keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                    nms_dets = dets_j[keep, :]

                nms_dets_j_all.append(nms_dets)

            overlaps_j_all = [[[] for u in range(len(nms_dets_j_all))] for v in range(len(nms_dets_j_all))]
            for j in range(len(nms_dets_j_all)):
                for k in range(j+1, len(nms_dets_j_all)):
                    overlaps = box_utils.bbox_overlaps(
                                    nms_dets_j_all[j][:, :-1].astype(dtype=np.float32, copy=False),
                                    nms_dets_j_all[k][:, :-1].astype(dtype=np.float32, copy=False))
                    overlaps_bin = overlaps.copy()
                    overlaps_bin[overlaps > tau] = 1
                    overlaps_bin[overlaps <= tau] = 0
                    overlaps_j_all[j][k] = overlaps_bin
            
            for j in range(len(nms_dets_j_all)):
                if cfg.OICR.CBL.RCNN_UseMTlayer_TYPE == "filter":
                    used_branch = cfg.OICR.CBL.RCNN_UseMTlayer_SELECTBRANCH
                    if j != used_branch:
                        continue
                for l in range(len(nms_dets_j_all[j])):
                    c_l = [nms_dets_j_all[j][l]]
                    if j < len(nms_dets_j_all) - 1:
                        for k in range(j+1, len(nms_dets_j_all)):
                            iou_bin_j_k = overlaps_j_all[j][k]
                            k_inds = np.where(iou_bin_j_k[l] == 1)[0]
                            if len(k_inds) > 0:
                                nms_dets_j_k_sel = nms_dets_j_all[k][k_inds]
                                max_ind_sel = np.argmax(nms_dets_j_k_sel[:, -1])
                                c_l.append(nms_dets_j_k_sel[max_ind_sel])
                    box_clusters.append(c_l)

            for box_c in box_clusters:
                box_c = np.vstack(box_c)
                boxes_in_c = box_c[:, :-1]
                scores_in_c = box_c[:, -1]
                ens_boxes = boxes_in_c[used_branch]
                stop_step = cfg.OICR.CBL.RCNN_UseMTlayer_STOPITER * cfg.SOLVER.MAX_ITER
                if cfg.OICR.CBL.RCNN_UseMTlayer_SCORE_GAMMA_TYPE == "sta":
                    if vis_needed["step"] <= stop_step:
                        gamma = cfg.OICR.CBL.RCNN_UseMTlayer_SCORE_GAMMA
                    else:
                        gamma = 0
                if cfg.OICR.CBL.RCNN_UseMTlayer_SCORE_ENSTYPE == 'add_lin':
                    add_v = cfg.OICR.CBL.RCNN_UseMTlayer_SCORE_ADDLIN_VALUE
                    ens_scores = scores_in_c[used_branch] * (add_v + pow(len(scores_in_c) / len(nms_dets_j_all), gamma))
                else:
                    ens_scores = scores_in_c[used_branch] * pow(len(scores_in_c) / len(nms_dets_j_all), gamma)
                ens_scores = ens_scores.astype(cls_prob.dtype)
                ### selected boxes
                gt_boxes = np.vstack((gt_boxes, ens_boxes))
                gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32)))
                gt_scores = np.vstack((gt_scores,
                    ens_scores.reshape(-1, 1) ))  # * np.ones((1, 1), dtype=np.float32)))

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores,
                 'gt_indices': None
                 }

    return proposals

def _sample_rois(all_rois, proposals, num_classes, reg_boxes=None, info_dict=None):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    t1 = time.time()
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    gt_indices = proposals['gt_indices']

    overlaps = box_utils.bbox_overlaps(
        all_rois.astype(dtype=np.float32, copy=False),
        gt_boxes.astype(dtype=np.float32, copy=False))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]

    fg_thresh = cfg.TRAIN.FG_THRESH
    bg_thresh_hi = cfg.TRAIN.FG_THRESH
    if info_dict["pseudo_mil"] == True:
        if cfg.OICR.CBL.IoUThresh == "linear":
            start_ratio = cfg.OICR.CBL.IoUThresh_START
            stop_ratio = cfg.OICR.CBL.IoUThresh_STOP
            fg_thresh = chg_ratio_linear_ori(info_dict["step"], start_ratio, stop_ratio, \
                                            stop_step=info_dict["all_step"], gamma=cfg.OICR.CBL.IoUThresh_GAMMA)
            info_dict["use_multi_range"] = False
            info_dict["multi_range_type"] = None

        elif cfg.OICR.CBL.IoUThresh == "static":
            fg_thresh = cfg.OICR.CBL.IoUThresh_Value
        elif cfg.OICR.CBL.IoUThresh == "linear_inv":
            fg_thresh = chg_ratio_linear_ori(info_dict["step"], 1.0, 0.5, \
                                            stop_step=info_dict["all_step"],
                                            gamma=cfg.OICR.CBL.IoUThresh_GAMMA)
        bg_thresh_hi = cfg.OICR.CBL.IoUThresh_NEG_Value

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < bg_thresh_hi)[0]
    ig_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH)[0]
    
    cls_loss_weights[ig_inds] = 0.0

    inds_info = {
        "gt_indices": gt_indices,
        "fg_inds": fg_inds,
        "gt_assignment": gt_assignment,
    }

    if (info_dict["pseudo_mil"] == True) and (not cfg.OICR.CBL.IoUThresh_NoIgn):
        ig_inds_1 = np.where((max_overlaps >= bg_thresh_hi) & (max_overlaps < fg_thresh))[0]
        cls_loss_weights[ig_inds_1] = 0.0
        if cfg.OICR.CBL.IoUThresh_IgnBG:
            labels[ig_inds_1] = 0

    labels[bg_inds] = 0

    real_labels = np.zeros((labels.shape[0], 21))
    for i in range(labels.shape[0]):
        real_labels[i, labels[i]] = 1

    if info_dict["pseudo_mil"] == True:
        real_labels = np.zeros((labels.shape[0], 21))
        cls_prob = info_dict["cls_prob"]
        for i in range(labels.shape[0]):
            real_labels[i, labels[i]] = cls_prob[i, labels[i]]
        cls_loss_weights_cls = np.ones((real_labels.shape[0], real_labels.shape[1]-1))
        for label, score in zip(gt_labels, gt_scores):
            cls_loss_weights_cls[:, label-1] = score

        inds_info["cls_loss_weights_onlycls"] = torch.tensor(cls_loss_weights_cls[0, :]).cuda().detach()
        cls_loss_weights_cls[ig_inds, :] = 0.0
        cls_loss_weights_cls[ig_inds_1, :] = 0.0
        cls_loss_weights_cls = cls_loss_weights_cls.astype(gt_scores.dtype)
        inds_info["cls_loss_weights_cls"] = torch.tensor(cls_loss_weights_cls).cuda().detach()


    if cfg.OICR.Need_Reg:
        # regression
        all_rois = torch.tensor(all_rois)
        gt_rois = torch.tensor(gt_boxes[gt_assignment, :])
        labels = torch.tensor(labels)
        bbox_target_data = _compute_targets_pytorch(all_rois, gt_rois)
        bbox_targets, bbox_inside_weights = \
            _get_bbox_regression_labels_pytorch(bbox_target_data, labels, num_classes)

        return real_labels, cls_loss_weights, labels, bbox_targets, bbox_inside_weights, inds_info

    else:
        return real_labels, cls_loss_weights


class OICRLosses(nn.Module):
    def __init__(self):
        super(OICRLosses, self).__init__()

    def forward(self, prob, labels_ic, cls_loss_weights, eps = 1e-6):
        loss = (labels_ic * torch.log(prob + eps))
        loss = loss.sum(dim=1)
        loss = -cls_loss_weights * loss
        ret = loss.sum() / loss.numel()
        return ret

def KLDivergenceLoss_2(x, y, mask=None, T=1):
    T1 = T

    if mask is not None:
        if mask.sum() > 0:
            p = F.log_softmax(x[mask]/T1, dim=-1)
            q = F.softmax(y[mask]/T, dim=-1)
            l_kl = F.kl_div(p, q, reduce=False)
            loss = torch.sum(l_kl)
            loss = loss / mask.sum().float()
        else:
            loss = 0.0
    else:
        p = F.log_softmax(x/T1, dim=-1)
        q = F.softmax(y/T, dim=-1)
        l_kl = F.kl_div(p, q, reduce=False)
        loss = l_kl.sum() / l_kl.size(0)
    
    return loss * T * T1

def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, cls_loss_ws, sigma=1.0, dim=[1], bg_balance=False):

    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    # out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = in_loss_box
    for i in sorted(dim, reverse=True):
      loss_box = loss_box.sum(i)
    loss_box = cls_loss_ws * loss_box
    if not bg_balance:
        loss_box = loss_box.mean()
    else:
        valid_num = len(torch.nonzero(cls_loss_ws))
        loss_box = loss_box.sum() / valid_num
    return loss_box


def chg_ratio_linear_ori(cur_step,
                            start_ratio,
                            stop_ratio=0.0,
                            start_step=0.0,
                            stop_step=cfg.SOLVER.MAX_ITER,
                            gamma=1.0
                            ):
    if cur_step <= stop_step and cur_step >= start_step:
        k = (cur_step - start_step) / (stop_step - start_step)
        cur_ratio = (stop_ratio - start_ratio) * pow(k, gamma) + start_ratio
    elif cur_step < start_step:
        cur_ratio = start_ratio
    else:
        cur_ratio = stop_ratio
    return cur_ratio

