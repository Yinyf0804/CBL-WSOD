import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils


class dilated_conv5_body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True))

        self.dim_out = 512

        self.spatial_scale = 1. / 8.

        self._init_modules()

    def _init_modules(self):
        assert cfg.VGG.FREEZE_AT in [0, 2, 3, 4, 5]
        for i in range(1, cfg.VGG.FREEZE_AT + 1):
            freeze_params(getattr(self, 'conv%d' % i))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'conv1.0.weight': 'conv1_0_w',
            'conv1.0.bias': 'conv1_0_b',
            'conv1.2.weight': 'conv1_2_w',
            'conv1.2.bias': 'conv1_2_b',
            'conv2.0.weight': 'conv2_0_w',
            'conv2.0.bias': 'conv2_0_b',
            'conv2.2.weight': 'conv2_2_w',
            'conv2.2.bias': 'conv2_2_b',
            'conv3.0.weight': 'conv3_0_w',
            'conv3.0.bias': 'conv3_0_b',
            'conv3.2.weight': 'conv3_2_w',
            'conv3.2.bias': 'conv3_2_b',
            'conv3.4.weight': 'conv3_4_w',
            'conv3.4.bias': 'conv3_4_b',
            'conv4.0.weight': 'conv4_0_w',
            'conv4.0.bias': 'conv4_0_b',
            'conv4.2.weight': 'conv4_2_w',
            'conv4.2.bias': 'conv4_2_b',
            'conv4.4.weight': 'conv4_4_w',
            'conv4.4.bias': 'conv4_4_b',
            'conv5.0.weight': 'conv5_0_w',
            'conv5.0.bias': 'conv5_0_b',
            'conv5.2.weight': 'conv5_2_w',
            'conv5.2.bias': 'conv5_2_b',
            'conv5.4.weight': 'conv5_4_w',
            'conv5.4.bias': 'conv5_4_b',
        }
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.VGG.FREEZE_AT + 1, 6):
            getattr(self, 'conv%d' % i).train(mode)

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, 'conv%d' % i)(x)
        return x


class dilated_conv5_body_norelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=2, dilation=2, bias=True))

        self.dim_out = 512

        self.spatial_scale = 1. / 8.

        self._init_modules()

    def _init_modules(self):
        assert cfg.VGG.FREEZE_AT in [0, 2, 3, 4, 5]
        for i in range(1, cfg.VGG.FREEZE_AT + 1):
            freeze_params(getattr(self, 'conv%d' % i))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'conv1.0.weight': 'conv1_0_w',
            'conv1.0.bias': 'conv1_0_b',
            'conv1.2.weight': 'conv1_2_w',
            'conv1.2.bias': 'conv1_2_b',
            'conv2.0.weight': 'conv2_0_w',
            'conv2.0.bias': 'conv2_0_b',
            'conv2.2.weight': 'conv2_2_w',
            'conv2.2.bias': 'conv2_2_b',
            'conv3.0.weight': 'conv3_0_w',
            'conv3.0.bias': 'conv3_0_b',
            'conv3.2.weight': 'conv3_2_w',
            'conv3.2.bias': 'conv3_2_b',
            'conv3.4.weight': 'conv3_4_w',
            'conv3.4.bias': 'conv3_4_b',
            'conv4.0.weight': 'conv4_0_w',
            'conv4.0.bias': 'conv4_0_b',
            'conv4.2.weight': 'conv4_2_w',
            'conv4.2.bias': 'conv4_2_b',
            'conv4.4.weight': 'conv4_4_w',
            'conv4.4.bias': 'conv4_4_b',
            'conv5.0.weight': 'conv5_0_w',
            'conv5.0.bias': 'conv5_0_b',
            'conv5.2.weight': 'conv5_2_w',
            'conv5.2.bias': 'conv5_2_b',
            'conv5.4.weight': 'conv5_4_w',
            'conv5.4.bias': 'conv5_4_b',
        }
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.VGG.FREEZE_AT + 1, 6):
            getattr(self, 'conv%d' % i).train(mode)

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, 'conv%d' % i)(x)
        return x


class Conv5_body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                             padding=1, bias=True),
                                   nn.ReLU(inplace=True))

        self.dim_out = 512

        self.spatial_scale = 1. / 8.

        self._init_modules()

    def _init_modules(self):
        assert cfg.VGG.FREEZE_AT in [0, 2, 3, 4, 5]
        for i in range(1, cfg.VGG.FREEZE_AT + 1):
            freeze_params(getattr(self, 'conv%d' % i))

    def detectron_weight_mapping(self):
        mapping_to_detectron = {
            'conv1.0.weight': 'conv1_0_w',
            'conv1.0.bias': 'conv1_0_b',
            'conv1.2.weight': 'conv1_2_w',
            'conv1.2.bias': 'conv1_2_b',
            'conv2.0.weight': 'conv2_0_w',
            'conv2.0.bias': 'conv2_0_b',
            'conv2.2.weight': 'conv2_2_w',
            'conv2.2.bias': 'conv2_2_b',
            'conv3.0.weight': 'conv3_0_w',
            'conv3.0.bias': 'conv3_0_b',
            'conv3.2.weight': 'conv3_2_w',
            'conv3.2.bias': 'conv3_2_b',
            'conv3.4.weight': 'conv3_4_w',
            'conv3.4.bias': 'conv3_4_b',
            'conv4.0.weight': 'conv4_0_w',
            'conv4.0.bias': 'conv4_0_b',
            'conv4.2.weight': 'conv4_2_w',
            'conv4.2.bias': 'conv4_2_b',
            'conv4.4.weight': 'conv4_4_w',
            'conv4.4.bias': 'conv4_4_b',
            'conv5.0.weight': 'conv5_0_w',
            'conv5.0.bias': 'conv5_0_b',
            'conv5.2.weight': 'conv5_2_w',
            'conv5.2.bias': 'conv5_2_b',
            'conv5.4.weight': 'conv5_4_w',
            'conv5.4.bias': 'conv5_4_b',
        }
        orphan_in_detectron = []

        return mapping_to_detectron, orphan_in_detectron

    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(cfg.VGG.FREEZE_AT + 1, 6):
            getattr(self, 'conv%d' % i).train(mode)

    def forward(self, x):
        for i in range(1, 6):
            x = getattr(self, 'conv%d' % i)(x)
        return x


class roi_2mlp_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, return_pool=False, return_fc6=False, xav_init=False):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096
        self.return_pool = return_pool
        self.return_fc6 = return_fc6

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if xav_init:
            import utils.weight_init as weight_init 
            for layer in [self.fc1, self.fc2]:
                weight_init.c2_xavier_fill(layer)

        # self._init_modules()

    # def _init_modules(self):

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rois, return_pool=False, return_fc6=False):
        x = self.roi_xform(
            x, rois,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        pool_feat = x
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x_fc6 = x
        x = F.relu(self.fc2(x), inplace=True)
        # print(self.return_fc6)
        if self.return_pool or return_pool:
            return x, pool_feat
        if self.return_fc6 or return_fc6:
            return x, x_fc6
        return x


class roi_2mlp_conv_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, return_pool=False, return_fc6=False):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096
        self.return_pool = return_pool
        self.return_fc6 = return_fc6

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # self.conv1 = nn.Conv2d(dim_in, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(dim_in, hidden_dim, kernel_size=1, bias=True)
        # self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

        # self._init_modules()

    # def _init_modules(self):

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b',
            'conv1.weight': 'roi_conv1_w',
            'conv1.bias': 'roi_conv1_b',
            # 'conv2.weight': 'roi_conv2_w',
            # 'conv2.bias': 'roi_conv2_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rois):
        x = self.roi_xform(
            x, rois,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        pool_feat = x
        batch_size = x.size(0)
        x1 = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x_fc6 = x1
        x1 = F.relu(self.fc2(x), inplace=True)

        x2 = F.relu(self.conv1(x), inplace=True)
        # x2 = F.relu(self.conv2(x), inplace=True)
        x2 = self.pool(x2).view(batch_size, -1)
        # print(self.return_fc6)
        if self.return_pool:
            return [x1, x2], pool_feat
        if self.return_fc6:
            return [x1, x2], x_fc6
        return [x1, x2]


class roi_2mlp_head_wtype(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rois=None, utype=0):
        if utype == 1:
            x = self.roi_xform(
                x, rois,
                method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
                resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
                spatial_scale=self.spatial_scale,
                sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
            )
            return x
        elif utype == 2:
            batch_size = x.size(0)
            x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
            x = F.relu(self.fc2(x), inplace=True)
            return x


class roi_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, roi_size=None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = hidden_dim = 4096
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.resolution = roi_size

        if roi_size is not None:
            self.roi_size = roi_size
        else:
            self.roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    def forward(self, x, rois):
        x = self.roi_xform(
            x, rois,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=self.roi_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        return x


class mlp_head(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = hidden_dim = 4096

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x


def freeze_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    for p in m.parameters():
        p.requires_grad = False


class cam_head(nn.Module):
    def __init__(self, dim_in, classes=20):
        super().__init__()
        self.dim_in = dim_in
        self.classes = classes

        self.conv6 = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1, padding=1, bias=True),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv7 = nn.Conv2d(dim_in, classes, 1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self._init_modules()

    # def _init_modules(self):

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'conv6.0.weight': 'conv6_0_w',
            'conv6.0.bias': 'conv6_0_b',
            'conv7.weight': 'conv7_0_w',
            'conv7.bias': 'conv7_0_b',
        }
        return detectron_weight_mapping, []

    def forward(self, x):
        if self.training:
            x = self.forward_train(x)
        else:
            x = self.forward_test(x)
        return x

    def forward_train(self, x):
        # return [N, C, H, W]
        x = self.conv6(x)
        x = self.avgpool(x)
        x = self.conv7(x)
        x = x.view(-1, self.classes)

        return x

    def forward_test(self, x):
        # return [N, C, H, W]
        x = self.conv6(x)
        x = F.conv2d(x, self.conv7.weight)
        x = F.relu(x)

        return x

class roi_2mlp_head_par(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, return_pool=False, return_fc6=False):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = 4096
        self.return_pool = return_pool
        self.return_fc6 = return_fc6

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # self._init_modules()

    # def _init_modules(self):

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b',
            'fc3.weight': 'fc8_w',
            'fc3.bias': 'fc8_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rois, return_pool=False):
        x = self.roi_xform(
            x, rois,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        pool_feat = x
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x_1 = F.relu(self.fc2(x), inplace=True)
        x_2 = F.relu(self.fc3(x), inplace=True)
        return x_1, x_2



class roi_db_2mlp_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale, use_db=True, return_pool=False, return_fc6=False, xav_init=False, dropout=False):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.use_db = use_db
        if self.use_db:
            self.hidden_dim = 2048
            self.dim_out = 4096
        else:
            self.hidden_dim = 4096
            self.dim_out = 4096
        self.return_pool = return_pool
        self.return_fc6 = return_fc6
        self.use_dropout = dropout

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        if use_db:
            from modeling.drop_block import DropBlock2D
            self.dropblock = DropBlock2D(block_size=3, drop_prob=0.3)
        if self.use_dropout:
            self.dropout = nn.Dropout()
        self.db_fc1 = nn.Linear(dim_in * roi_size**2, self.hidden_dim)
        self.db_fc2 = nn.Linear(self.hidden_dim, self.dim_out)

        if xav_init:
            import utils.weight_init as weight_init 
            for layer in [self.fc1, self.fc2]:
                weight_init.c2_xavier_fill(layer)

        # self._init_modules()

    # def _init_modules(self):

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'db_fc1.weight': 'db_fc6_w',
            'db_fc1.bias': 'db_fc6_b',
            'db_fc2.weight': 'db_fc7_w',
            'db_fc2.bias': 'db_fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rois, return_pool=False, return_fc6=False):
        x = self.roi_xform(
            x, rois,
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        pool_feat = x
        batch_size = x.size(0)

        if self.use_db:
            x = self.dropblock(x)
        x = F.relu(self.db_fc1(x.view(batch_size, -1)), inplace=True)
        if self.dropout:
            x = self.dropout(x)
        x_fc6 = x
        x = F.relu(self.db_fc2(x), inplace=True)
        if self.dropout:
            x = self.dropout(x)
        # print(self.return_fc6)
        if self.return_pool or return_pool:
            return x, pool_feat
        if self.return_fc6 or return_fc6:
            return x, x_fc6
        return x