from functools import wraps
import importlib
import logging
import copy
import numpy as np
import os
import cv2
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.pcl.oicr_cbl import OICR, OICRLosses, _smooth_l1_loss, KLDivergenceLoss_2
from model.pcl_losses.functions.pcl_losses import PCLLosses
from model.regression.bbox_transform import bbox_transform_inv
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.pcl_heads as pcl_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.boxes as box_utils
import utils.vgg_weights_helper as vgg_utils

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        if cfg.FAST_RCNN.ROI_BOX_HEAD == "vgg16.roi_db_2mlp_head":
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale, use_db=cfg.FAST_RCNN.DB_ON, dropout=cfg.FAST_RCNN.DB_DROPOUT)
        else:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)

        self.Box_MIL_Outs = pcl_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = pcl_heads.refine_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
        self.RCNN_Cls_Reg = pcl_heads.cls_regress_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Refine_Losses = [OICRLosses() for i in range(cfg.REFINE_TIMES)]
        self.Cls_Loss = OICRLosses()

        if cfg.OICR.CBL.MeanTeacher_FC:
            self.Refine_EMA = copy.deepcopy(self.Box_Refine_Outs.refine_score[-1])
            self.Refine_EMA_Loss = OICRLosses()
            for param in self.Refine_EMA.parameters():
                param.requires_grad = False
        elif cfg.OICR.CBL.MeanTeacher_ROIFC:
            self.Box_Head_EMA = copy.deepcopy(self.Box_Head)
            self.Refine_EMA = copy.deepcopy(self.Box_Refine_Outs.refine_score[-1])
            self.Refine_EMA_Loss = OICRLosses()
            for param in self.Box_Head_EMA.parameters():
                param.requires_grad = False
            for param in self.Refine_EMA.parameters():
                param.requires_grad = False
        elif cfg.OICR.CBL.MeanTeacher_ALL:
            self.Conv_Body_EMA = copy.deepcopy(self.Conv_Body)
            self.Box_Head_EMA = copy.deepcopy(self.Box_Head)
            if cfg.OICR.CBL.REFINE_INIT_WEIGHT == "cls":
                self.Refine_EMA = copy.deepcopy(self.RCNN_Cls_Reg.cls_score)
            else:
                self.Refine_EMA = copy.deepcopy(self.Box_Refine_Outs.refine_score[-1])
            self.Refine_EMA_Loss = OICRLosses()
            for param in self.Conv_Body_EMA.parameters():
                param.requires_grad = False
            for param in self.Box_Head_EMA.parameters():
                param.requires_grad = False
            for param in self.Refine_EMA.parameters():
                param.requires_grad = False

        self.mean_teacher = cfg.OICR.CBL.MeanTeacher_FC or cfg.OICR.CBL.MeanTeacher_ALL or cfg.OICR.CBL.MeanTeacher_ROIFC
            
        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def update_ema_variables(self, alpha, global_step):
        if cfg.OICR.CBL.MeanTeacher_FC:
            models, ema_models = [self.Box_Refine_Outs.refine_score[-1]], [self.Refine_EMA]
        elif cfg.OICR.CBL.MeanTeacher_ROIFC:
            models, ema_models = [self.Box_Head, self.Box_Refine_Outs.refine_score[-1]], \
                                 [self.Box_Head_EMA, self.Refine_EMA]
        else:
            models, ema_models = [self.Conv_Body, self.Box_Head, \
                                    self.Box_Refine_Outs.refine_score[-1]], \
                                 [self.Conv_Body_EMA, self.Box_Head_EMA, self.Refine_EMA]
            if cfg.OICR.CBL.MeanTeacher_UPDATE_MULTI == 'cls':
                models = [self.Conv_Body, self.Box_Head, \
                                    self.RCNN_Cls_Reg.cls_score]

        if cfg.OICR.CBL.MeanTeacher_UPDATE == "multi":
            if cfg.OICR.CBL.MeanTeacher_UPDATE_MULTI == 'last_cls':
                models_fc = [self.Box_Refine_Outs.refine_score[-1], self.RCNN_Cls_Reg.cls_score]
            elif cfg.OICR.CBL.MeanTeacher_UPDATE_MULTI in ['refine_cls','refine_avg_cls','refine_avg_cls_lin']:
                models_fc = [m for m in self.Box_Refine_Outs.refine_score]
                models_fc.append(self.RCNN_Cls_Reg.cls_score)
            else:
                models_fc = self.Box_Refine_Outs.refine_score
            ema_models_fc = self.Refine_EMA
            if cfg.OICR.CBL.MeanTeacher_UPDATE_METHOD == 'cos':
                PI = 3.1416
                total_step = cfg.SOLVER.MAX_ITER
                alpha = 1 - (1- alpha) * (np.cos(PI * global_step / total_step) + 1) / 2
            else:
                alpha = min(1 - 1 / (global_step + 1), alpha)
            if cfg.OICR.CBL.MeanTeacher_UPDATE_MULTI == 'refine_avg_cls':
                beta = (1 - alpha) / (len(models_fc) - 1)
                betas = [beta / 2 for _ in range(len(models_fc) - 1)]
                betas.append((1 - alpha) / 2)
            else:
                beta = (1 - alpha) / len(models_fc)
                betas = [beta for _ in range(len(models_fc))]
            for ind, model_fc in enumerate(models_fc):
                for ema_param, param in zip(ema_models_fc.parameters(), model_fc.parameters()):
                    beta = betas[ind]
                    if ind == 0:
                        ema_param.data.mul_(alpha).add_(beta, param.data)
                    else:
                        ema_param.data.add_(beta, param.data)
            models.pop(-1)
            ema_models.pop(-1)

        # Use the true average until the exponential average is more correct
        if len(models) > 0:
            if cfg.OICR.CBL.MeanTeacher_UPDATE_METHOD == 'cos':
                PI = 3.1416
                total_step = cfg.SOLVER.MAX_ITER
                alpha = 1 - (1- alpha) * (np.cos(PI * global_step / total_step) + 1) / 2
                # print(alpha)
            else:
                alpha = min(1 - 1 / (global_step + 1), alpha)
            for model, ema_model in zip(models, ema_models):
                for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                    ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def forward(self, data, rois, labels, step=0, indexes=0, im_scales=None, roi=None, step_scale=0.5):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels, step, indexes, im_scales, roi, step_scale)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels, step, indexes, im_scales, roi, step_scale)

    def _forward(self, data, rois, labels, step, indexes, im_scales, roi, step_scale):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        box_feat, pool_feat = self.Box_Head(blob_conv, rois, return_pool=True)
        mil_score, det_score = self.Box_MIL_Outs(box_feat, return_det=True)
        refine_score = self.Box_Refine_Outs(box_feat)
        if not self.training and cfg.OICR.CBL.Test_UseShareMT:
            refine_score_ema = self.Refine_EMA(box_feat)
            refine_score_ema = F.softmax(refine_score_ema, dim=1)
        else:
            if cfg.OICR.CBL.MeanTeacher_FC:
                refine_score_ema = self.Refine_EMA(box_feat)
                refine_score_ema = F.softmax(refine_score_ema, dim=1)
            elif cfg.OICR.CBL.MeanTeacher_ROIFC:
                blob_conv_ema = blob_conv.clone().detach()
                box_feat_ema = self.Box_Head_EMA(blob_conv_ema, rois)
                refine_score_ema = self.Refine_EMA(box_feat_ema)
                refine_score_ema = F.softmax(refine_score_ema, dim=1)
            elif cfg.OICR.CBL.MeanTeacher_ALL:
                blob_conv_ema = self.Conv_Body_EMA(im_data)
                box_feat_ema, pool_feat_ema = self.Box_Head_EMA(blob_conv_ema, rois, return_pool=True)
                refine_score_ema = self.Refine_EMA(box_feat_ema)
                refine_score_ema = F.softmax(refine_score_ema, dim=1)

        cls_score, bbox_pred = self.RCNN_Cls_Reg(box_feat)

        if cfg.OICR.CBL.MT_REFINE:
            mt_refine_score = self.Refine_EMA(box_feat)
            mt_refine_score = F.softmax(mt_refine_score, dim=1)

        if self.training:
            rois_n = rois[:, 1:]
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(-1, 4 * (cfg.MODEL.NUM_CLASSES + 1))
            pred_boxes = bbox_transform_inv(rois_n, box_deltas, 1)
            im_shape = data.shape[-2:]
            pred_boxes = box_utils.clip_boxes_2(pred_boxes, im_shape)

            return_dict['losses'] = {}

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            loss_im_cls_ori = loss_im_cls.clone().detach()

            start_step = cfg.SOLVER.MAX_ITER * cfg.OICR.CBL.LOSSWEIGHT_STARTITER
            if step >= start_step:
                if cfg.OICR.CBL.LOSSWEIGHT == "linear":
                    stop_ratio = cfg.OICR.CBL.LOSSWEIGHT_STOPRATIO
                    loss_weight_mil = self.chg_ratio_linear_ori(step, 1.0, stop_ratio, start_step=start_step, stop_step=cfg.OICR.CBL.EXTRA_ITER*step_scale)
                    loss_im_cls = loss_im_cls * loss_weight_mil
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            info_dict = {"step": step, "pseudo_mil": False, "mt_scores_sup": refine_score_ema, "all_step": cfg.OICR.CBL.EXTRA_ITER*step_scale}
            # refinement loss
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]

            pcl_outputs = []
            for i_refine, refine in enumerate(refine_score):
                info_dict["i_refine"] = i_refine
                if i_refine == 0:
                    pcl_output = OICR(boxes, mil_score, im_labels, refine, pred_boxes, info_dict)
                    pcl_output_0 = pcl_output
                else:
                    pcl_output = OICR(boxes, refine_score[i_refine - 1],
                                    im_labels, refine, pred_boxes, info_dict)
                pcl_outputs.append(pcl_output)
                
                refine_loss = self.Refine_Losses[i_refine](refine, pcl_output['labels'],
                                                        pcl_output['cls_loss_weights'])
                if i_refine == 0:
                    refine_loss = refine_loss * cfg.OICR.Weight_Firstbranch

                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()

            if self.mean_teacher:
                refine_loss_ema = self.Refine_EMA_Loss(refine_score_ema, pcl_output['labels'],
                                                        pcl_output['cls_loss_weights'])
                refine_loss_ema = refine_loss_ema.detach()
                return_dict['losses']['refine_loss_ema'] = refine_loss_ema.clone()
            
            if step >= start_step:
                info_dict["pseudo_mil"] = True
                info_dict["mil_gt"] = pcl_output_0["proposals"]
                info_dict["mil_scores"] = mil_score

                # generate pseudo labels
                pcl_output_p = OICR(boxes, refine_score_ema, im_labels, cls_score, info_dict=info_dict)

                pseudo_labels = pcl_output_p['labels'][:, 1:]

                mil_pseudoloss = 0.0
                cls_loss_weights = pcl_output_p["inds_info"]['cls_loss_weights_onlycls']
                for c in torch.nonzero(labels[0, :])[:, 0]:
                    pseudo_labels_c = pseudo_labels[:, c]
                    mask_c = pseudo_labels_c > 0
                    mil_score_c = mil_score[:, c]
                    mil_pseudoloss_c = KLDivergenceLoss_2(mil_score_c, pseudo_labels_c, mask_c)
                    if cfg.OICR.CBL.CLS_WEIGHT:
                        mil_pseudoloss_c = mil_pseudoloss_c * cls_loss_weights[c].float()
                    mil_pseudoloss += mil_pseudoloss_c

                if cfg.OICR.CBL.LOSSWEIGHT == "linear":
                    loss_weight_mil_pseudo = 1 - loss_weight_mil
                    mil_pseudoloss = mil_pseudoloss * loss_weight_mil_pseudo

                if cfg.OICR.CBL.Pseudo_MIL:
                    return_dict['losses']['mil_pseudoloss'] = mil_pseudoloss.clone()

            # regression
            if cfg.OICR.Need_Reg:
                info_dict["pseudo_mil"] = False
                info_dict["i_refine"] = 3
                if cfg.OICR.CBL.RCNN_UseMTlayer:
                    if step >= cfg.OICR.CBL.RCNN_UseMTlayer_STARTITER * cfg.SOLVER.MAX_ITER:
                        multigt = cfg.OICR.CBL.RCNN_UseMTlayer_MULTIGT
                        refine_score_ema_used = refine_score + [refine_score_ema]
                        info_dict["score_ori"] = (refine_score_ema + refine_score[-1]) / 2
                        pcl_output = OICR(boxes, refine_score_ema_used,
                                            im_labels, cls_score, info_dict=info_dict, multigt=multigt)           
                    else:
                        refine_score_ema_used = refine_score[-1]
                        pcl_output = OICR(boxes, refine_score[-1],
                                        im_labels, cls_score, info_dict=info_dict)

                ### cls
                cls_loss_ws = pcl_output['cls_loss_weights']
                RCNN_loss_cls = self.Cls_Loss(cls_score, pcl_output['labels'], cls_loss_ws)

                ### reg
                rois_label = pcl_output['rois_labels']
                rois_target = pcl_output['bbox_targets']
                rois_inside_ws = pcl_output['bbox_inside_weights']

                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1)

                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, cls_loss_ws, bg_balance=False)

                return_dict['losses']['cls_loss'] = RCNN_loss_cls.clone()
                return_dict['losses']['reg_loss'] = RCNN_loss_bbox.clone()
                
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)

        else:
            # Testing
            refine_score.append(cls_score)
            return_dict['mil_score'] = mil_score
            
            if cfg.OICR.CBL.Test_UseMT:
                refine_score.append(refine_score_ema)

            if cfg.OICR.CBL.MT_REFINE_TEST:
                refine_score.append(mt_refine_score)

            if cfg.OICR.CBL.Test_UseMTOnly:
                refine_score = [refine_score_ema]

            if cfg.OICR.CBL.Test_UseMTMean:
                refine_score_ori = torch.stack(refine_score, dim=0).mean(dim=0)
                refine_score = [refine_score_ori, refine_score_ema]

            if cfg.OICR.CBL.Test_UseMTMean2:
                refine_score_ori = torch.stack(refine_score[:-1], dim=0).mean(dim=0)
                refine_score_cls = (refine_score[-1] + refine_score_ema) / 2
                refine_score = [refine_score_ori, refine_score_cls]
            
            if cfg.OICR.CBL.Test_UseMTMean3:
                refine_score_ori = torch.stack(refine_score[:-1], dim=0).mean(dim=0)
                refine_score = [refine_score_ori, refine_score[-1], refine_score_ema]

            return_dict['refine_score'] = refine_score
            if cfg.OICR.Need_Reg:
                rois = rois[:, 1:]
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(-1, 4 * (cfg.MODEL.NUM_CLASSES + 1))
                pred_boxes = bbox_transform_inv(rois, box_deltas, 1)
                im_shape = data.shape[-2:]
                pred_boxes = box_utils.clip_boxes_2(pred_boxes, im_shape)
                return_dict['rois'] = pred_boxes

        return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoICrop':
            grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                xform_out = F.max_pool2d(xform_out, 2, 2)
        elif method == 'RoIAlign':
            xform_out = RoIAlignFunction(
                resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    def chg_ratio_linear_ori(self, cur_step,
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

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    if hasattr(m_child, "detectron_weight_mapping"):
                        child_map, child_orphan = m_child.detectron_weight_mapping()
                        d_orphan.extend(child_orphan)
                        for key, value in child_map.items():
                            new_key = name + '.' + key
                            d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan
        self.mapping_to_detectron["Refine_EMA.weight"] = "Refine_EMA_weight"
        self.mapping_to_detectron["Refine_EMA.bias"] = "Refine_EMA_bias"
        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value