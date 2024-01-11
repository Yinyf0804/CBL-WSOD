from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
import pickle
from PIL import Image

import torch

import _init_paths
import utils.boxes as box_utils
from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder, model_builder_corpol, model_builder_oicr
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
from utils.io import save_object
from utils.timer import Timer

class VOC(object):
    def __init__(self, dataset_name, proposal_file=None):
        self.dataset_name = dataset_name
        self.voc = JsonDataset(dataset_name)
        self.roidb = self.voc.get_roidb(proposal_file=proposal_file)
        self.category_to_id_map = self.voc.category_to_id_map
        print(self.category_to_id_map)

    def show_gt(self):
        output_dir_gt = '/ghome/yinyf/wsddn/outpic/gt/{}/'.format(self.dataset_name)
        output_dir_ss = '/ghome/yinyf/wsddn/outpic/ss/{}/'.format(self.dataset_name)

        if not os.path.exists(output_dir_gt):
            os.makedirs(output_dir_gt)
        if not os.path.exists(output_dir_ss):
            os.makedirs(output_dir_ss)
        for entry in self.roidb:
            img_name = entry['image'].split('/')[-1].split('.')[0]
            self.voc._add_gt_annotations_withbox(entry)
            gt_classes = entry['gt_classes'][0]
            gt_cls = np.where(gt_classes == 0)[0]

            im = cv2.imread(entry['image'])
            gt_boxes = np.array([obj['clean_bbox'] for obj in entry['objects']])
            for box in gt_boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                color_pred = (0,255,255)
                cv2.rectangle(im, (x1, y1), (x2, y2), color_pred, 1)
            img_name_all = output_dir_gt + img_name + '.jpg'
            cv2.imwrite(img_name_all, im)

            im = cv2.imread(entry['image'])
            boxes = entry['boxes']
            for box in boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                color_pred = (0,255,255)
                cv2.rectangle(im, (x1, y1), (x2, y2), color_pred, 1)
            img_name_all = output_dir_ss + img_name + '.jpg'
            cv2.imwrite(img_name_all, im)

            print(img_name)
            print(img_name_all)
            print(gt_classes)
            input()

    def _find_cls_img(self, cls_name):
        cls_id = self.category_to_id_map[cls_name] - 1
        cls_imgs = []
        for entry in self.roidb:
            self.voc._add_gt_annotations(entry)
            gt_classes = entry['gt_classes'][0]
            if gt_classes[cls_id] == 1:
                img_name = entry['image'].split('/')[-1].split('.')[0]
                cls_imgs.append(img_name)
        print(cls_imgs)

    def _draw_ss_boxes(self, img_name_sel=None):
        output_dir = "/ghome/yinyf/wsddn/outpic/ss/"
        for entry in self.roidb:
            img_name = entry['image'].split('/')[-1].split('.')[0]
            if img_name_sel == img_name:
                img_path = entry['image']
                boxes = entry['boxes']
                self.voc._add_gt_annotations_withbox(entry)
                im = cv2.imread(img_path)

                gt_boxes = np.array([obj['clean_bbox'] for obj in entry['objects']])
                overlaps = box_utils.bbox_overlaps(
                    boxes.astype(dtype=np.float32, copy=False),
                    gt_boxes.astype(dtype=np.float32, copy=False))
                overlaps = overlaps.max(axis=1)
                box_sorted_ind = np.argsort(overlaps)[::-1][:200]

                for ind in box_sorted_ind:
                    box = boxes[ind]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    color_pred = (0,255,255)
                    cv2.rectangle(im, (x1, y1), (x2, y2), color_pred, 1)
                img_name = output_dir + img_name_sel + '.jpg'
                cv2.imwrite(img_name, im)

    def make_new_propsals(self, proposal_file, sav_num, output_file):
        with open(proposal_file, 'rb') as f:
            proposals = pickle.load(f)
        proposals_ori = proposals.copy()  
        for i, entry in enumerate(self.roidb):
            self.voc._add_gt_annotations_withbox(entry)
            gt_boxes = np.array([obj['clean_bbox'] for obj in entry['objects']])
            boxes = proposals_ori['boxes'][i]
            overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False))
            overlaps = overlaps.max(axis=1)
            THRESH = 0.1
            # THRESH = 0
            bg_ind = np.where(overlaps<=THRESH)[0]
            fg_ind = np.where(overlaps>THRESH)[0]
            fg_num = len(fg_ind)
            bg_num_need = np.max(sav_num - fg_num, 0)
            # print(bg_num_need, len(bg_ind), fg_num, len(boxes))
            # input()
            if bg_num_need < len(bg_ind):
                np.random.shuffle(bg_ind)
                bg_ind_rand = bg_ind[:bg_num_need]
                ind_all = np.hstack((fg_ind, bg_ind_rand))
                boxes_remain = boxes[ind_all, :]
                proposals['boxes'][i] = boxes_remain
        

        output_file = output_file.format(sav_num, int(THRESH*100))
        with open(output_file, 'wb') as f:
            pickle.dump(proposals, f)
        print('Saved in {}'.format(output_file))


if __name__ == "__main__":
    # dataset_name = "voc_2007_test"
    dataset_name = "voc_2007_trainval"
    # dataset_name = "voc_2012_train"
    # dataset_name = "voc_2012_val"
    # dataset_name = "voc_2012_trainval"

    proposal_file = '/ghome/yinyf/pcl/data/selective_search_data/{}.pkl'.format(dataset_name)
    # proposal_file = '/ghome/yinyf/pcl/data/selective_search_data/voc_2007_test.pkl'
    VOC = VOC(dataset_name, proposal_file)
    output_file = '/ghome/yinyf/pcl/data/selective_search_data/voc_2007_trainval_{}_{}e-2.pkl'
    VOC.make_new_propsals(proposal_file, 2000, output_file)
    # VOC.show_gt()
    # VOC._find_cls_img('dog')
    # VOC._draw_ss_boxes('006716')
