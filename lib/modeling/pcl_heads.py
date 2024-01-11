import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
# from model.py_utils import TopPool, BottomPool, LeftPool, RightPool

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np
import math


class mil_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mil_score0 = nn.Linear(dim_in, dim_out)
        self.mil_score1 = nn.Linear(dim_in, dim_out)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.mil_score0.weight, std=0.01)
        init.constant_(self.mil_score0.bias, 0)
        init.normal_(self.mil_score1.weight, std=0.01)
        init.constant_(self.mil_score1.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'mil_score0.weight': 'mil_score0_w',
            'mil_score0.bias': 'mil_score0_b',
            'mil_score1.weight': 'mil_score1_w',
            'mil_score1.bias': 'mil_score1_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x, mask0=None, mask1=None, return_det=False):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        mil_score0 = self.mil_score0(x)
        mil_score1 = self.mil_score1(x)
        if mask0 is not None:
            mil_score0 = mil_score0 * mask0
        if mask1 is not None:
            mil_score1 = mil_score1 * mask1
        mil_score = F.softmax(mil_score0, dim=0) * F.softmax(mil_score1, dim=1)

        if return_det:
            return mil_score, F.softmax(mil_score0, dim=0)

        return mil_score


class refine_outputs(nn.Module):
    def __init__(self, dim_in, dim_out, s_ver=False):
        super().__init__()
        self.refine_score = []
        for i_refine in range(cfg.REFINE_TIMES):
            self.refine_score.append(nn.Linear(dim_in, dim_out))
        self.refine_score = nn.ModuleList(self.refine_score)
        self.s_ver = s_ver

        self._init_weights()

    def _init_weights(self):
        for i_refine in range(cfg.REFINE_TIMES):
            init.normal_(self.refine_score[i_refine].weight, std=0.01)
            init.constant_(self.refine_score[i_refine].bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {}
        for i_refine in range(cfg.REFINE_TIMES):
            detectron_weight_mapping.update({
                'refine_score.%d.weight' % i_refine: 'refine_score%d_w' % i_refine,
                'refine_score.%d.bias' % i_refine: 'refine_score%d_b' % i_refine
            })
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        refine_score = [F.softmax(refine(x), dim=1) for refine in self.refine_score]
        if self.s_ver:
            refine_score_ver = [F.softmax(refine(x), dim=0) for refine in self.refine_score]
            return refine_score, refine_score_ver
        else:
            return refine_score

class cls_regress_outputs(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.cls_score = nn.Linear(dim_in, dim_out)
        self.reg_score = nn.Linear(dim_in, 4*dim_out)

        self._init_weights()

    def _init_weights(self):
        init.normal_(self.cls_score.weight, std=0.01)
        init.constant_(self.cls_score.bias, 0)
        init.normal_(self.reg_score.weight, std=0.001)
        init.constant_(self.reg_score.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'reg_score.weight': 'reg_score_w',
            'reg_score.bias': 'reg_score_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = F.softmax(self.cls_score(x), dim=1)
        reg_score = self.reg_score(x)
        return cls_score, reg_score


def mil_losses(cls_score, labels):
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    return loss.mean()


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
