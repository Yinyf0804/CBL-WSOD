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
import matplotlib.pyplot as plt 

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

Classes = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

def boxes_nms(dataset_name, detect_file, detect_nms_file, detect_bnms_file, Need_Reg=True, replace=None):
    with open(detect_file, 'rb') as f:
        all_boxes = pickle.load(f)["all_boxes"]
    if not os.path.exists(detect_nms_file):
        dataset = JsonDataset(dataset_name)
        roidb = dataset.get_roidb()
        num_images = len(roidb)
        num_classes = cfg.MODEL.NUM_CLASSES + 1
        start_boxes = empty_results(num_images)
        final_boxes = empty_results(num_images)
        for i, entry in enumerate(roidb):
            # print(entry['image'], list(all_boxes.keys())[0])
            if replace is not None:
                new_img = os.path.join(replace, entry['image'].split('/')[-1])
                boxes = all_boxes[new_img]
            else:
                boxes = all_boxes[entry['image']]
            # print(boxes['scores'], boxes['boxes'])
            _, _, cls_boxes_i = box_results_with_nms_and_limit(boxes['scores'],
                                                            boxes['boxes'], 
                                                            Need_Reg=Need_Reg)
            extend_results(i, final_boxes, cls_boxes_i)
            beforenms_boxes = get_beforenms_boxes(boxes['boxes'], boxes['scores'])
            start_boxes[i]= beforenms_boxes
        save_object({"all_boxes": final_boxes}, detect_nms_file)
        save_object({"all_boxes": start_boxes}, detect_bnms_file)
    else:
        with open(detect_nms_file, 'rb') as f:
            final_boxes = pickle.load(f)["all_boxes"]
        with open(detect_bnms_file, 'rb') as f:
            start_boxes = pickle.load(f)["all_boxes"]
        # start_boxes = []
    return start_boxes, final_boxes


def boxes_draw_target(dataset_name, all_boxes, draw_gt=False, iter_num=None, sav_dir=None, index=None, index_cls=None, draw_ok=True, replace=None, obj_num=1, same_color=False):
    THRESH = 0.3
    draw_gt = draw_gt
    # if draw_gt:
    #     sav_dir = "/ghome/yinyf/wsddn/outpic/mmdet_iter2_test/0%d/"%(THRESH*10)
    # else:
    #     sav_dir = "/ghome/yinyf/wsddn/outpic/mmdet_iter2_test/0%d_nogt/"%(THRESH*10)
    
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb()

    ok_image = []
    for i, entry in enumerate(roidb):
        dataset._add_gt_annotations_withbox(entry)
        img_path = entry['image']
        if index is not None:
            if not isinstance(index, list):
                index = [index]
            signal = 0
            for ind_n in index:
                if ind_n in img_path:
                    signal = 1
            if signal == 0:
                continue
        
        if index_cls is not None:
            gt_cls = np.where(entry['gt_classes'][0] == 1)[0]
            # print(gt_cls)
            gt_cls_name = [Classes[c] for c in gt_cls]
            if not isinstance(index_cls, list):
                index_cls = [index_cls]
            signal = 0
            for ind_n in index_cls:
                if ind_n in gt_cls_name:
                    signal = 1
            if signal == 0:
                continue

        boxes = all_boxes[i]
        im = cv2.imread(img_path)
        img_name = img_path.split('/')[-1].split('.')[0]
        
        gt_boxes = np.array([obj['clean_bbox'] for obj in entry['objects']])
        if draw_gt:
            if len(gt_boxes) > 0:
                for box in gt_boxes:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0,0,255), 5)

        if len(boxes) > 0:
            overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False))
            overlaps_1 = overlaps.max(axis=1)
            overlaps_ok = overlaps.copy()
            overlaps_ok[overlaps > 0.5] = 1
            overlaps_ok[overlaps <= 0.5] = 0
            overlaps_ok = overlaps_ok.sum(axis=0)
            ind_ok = np.where(overlaps_ok == 1)[0]
            # print(overlaps_ok, ind_ok, len(gt_boxes), len(boxes))
            if draw_ok:
                if (len(ind_ok) == len(gt_boxes)) and len(gt_boxes) > obj_num:
                    ok_image.append(img_name)

                    for ind, box in enumerate(boxes):
                        if box[4] > THRESH:
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            score = '%.2f'%(box[4])
                            color_pred = (0,255,0) if overlaps_1[ind] < 0.5 else (0,255,255)
                            # color_pred = (255,0,0)
                            cv2.rectangle(im, (x1, y1), (x2, y2), color_pred, 5)
                            # cv2.putText(im, score, (2, j*15), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,0,0), thickness=1)
                    img_name = sav_dir + img_path.split('/')[-1].split('.')[0] + '.jpg'
                    cv2.imwrite(img_name, im)
            else:
                h, w, _ = im.shape
                for ind, box in enumerate(boxes):
                    if box[4] > THRESH:
                    # if box[4] > 0.3 < box[4] < 0.5:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        x1, y1, x2, y2 = adjust_box(x1, y1, x2, y2, w, h)
                        score = '%.2f'%(box[4])
                        if same_color:
                            color_pred = (255,0,0)
                        else:
                            color_pred = (0,255,0) if overlaps_1[ind] < 0.5 else (0,255,255)
                        # color_pred = (200,100,0)
                        cv2.rectangle(im, (x1, y1), (x2, y2), color_pred, 5)
            # input()
                img_name = sav_dir + img_path.split('/')[-1].split('.')[0] + '.jpg'
                cv2.imwrite(img_name, im)
        else:
            if not draw_ok:
                img_name = sav_dir + img_path.split('/')[-1].split('.')[0] + '.jpg'
                cv2.imwrite(img_name, im)
            print(len(boxes), img_name)
        # print(img_name)
        if i % 1000 == 0:
            print("Have drawn {} / {} picture".format(i, len(roidb)))
    print(ok_image)

def combine_pics(pics, ini_dirs, sav_dir):
    if not os.path.exists(sav_dir):
        os.makedirs(sav_dir)
    for pic in pics:
        imgs = []
        for ini_dir in ini_dirs:
            img_path = os.path.join(ini_dir, pic)
            # print(img_path)
            imgs.append(np.array(cv2.imread(img_path)))

        fig = plt.figure()
        pic1 = fig.add_subplot(231)
        plt.axis('off')
        pic2 = fig.add_subplot(232)
        plt.axis('off')
        pic3 = fig.add_subplot(233)
        plt.axis('off')
        pic4 = fig.add_subplot(234)
        plt.axis('off')
        pic5 = fig.add_subplot(235)
        plt.axis('off')
        pic6 = fig.add_subplot(236)
        plt.axis('off')
        pic1.imshow(imgs[0].astype(np.int32))
        pic2.imshow(imgs[1].astype(np.int32))
        pic3.imshow(imgs[2].astype(np.int32))
        pic4.imshow(imgs[3].astype(np.int32))
        pic5.imshow(imgs[4].astype(np.int32))
        pic6.imshow(imgs[5].astype(np.int32))
        plt.savefig(os.path.join(sav_dir, pic))


def filter_boxes(boxes_all):
    new_boxes_all = empty_results(len(boxes_all))
    for i, boxes in enumerate(boxes_all):
        # ind = np.where(boxes[:, 4] > 0.2)
        ind = np.where(boxes[:, 4] > 0.4)
        new_boxes_all[i] = boxes[ind]
    return new_boxes_all

def find_sur_boxes(t_boxes_all, boxes_all):
    sur_boxes_all = empty_results(len(boxes_all))
    for i, (t_boxes, boxes) in enumerate(zip(t_boxes_all, boxes_all)):
        if i == 100:
            break
        # sur_ind_all = np.array([])
        for t_box in t_boxes:
            sur_ind = np.where((boxes[:, 0] <= t_box[0]) & (boxes[:, 1] <= t_box[1]) &
                            (boxes[:, 2] >= t_box[2]) & (boxes[:, 3] >= t_box[3]))[0]
            sur_boxes = boxes[sur_ind]
            ind = np.argsort(sur_boxes[:, 1])
            sur_boxes_all[i].append(sur_boxes[ind])

        # sur_ind_all = np.unique(sur_ind_all).astype(int)
    return sur_boxes_all
        # areas = (sur_boxes[:, 2] - sur_boxes[:, 0]) * (sur_boxes[:, 3] - sur_boxes[:, 1])

def get_beforenms_boxes(boxes, scores):
    '''
    boxes: [N, 4]
    scores: [N, C]
    '''
    scores = scores[:, 1:]
    boxes_cls = scores.argmax(axis=1) + 1
    boxes_score = scores.max(axis=1)
    boxes_final = np.hstack((boxes, boxes_score[:, np.newaxis], boxes_cls[:, np.newaxis]))
    return boxes_final


def empty_results(num_images):
    all_boxes = [[] for _ in range(num_images)]
    return all_boxes


def extend_results(index, all_res, im_res):
    # Skip cls_idx 0 (__background__)
    boxes = []
    for cls_idx in range(1, len(im_res)):
        im_res_cls = im_res[cls_idx]
        cls_inds = cls_idx * np.ones(len(im_res_cls))
        im_res_cls = np.hstack((im_res_cls, cls_inds[:, np.newaxis]))
        if len(boxes) == 0:
            boxes = im_res_cls
        else:
            boxes = np.vstack((boxes, im_res_cls))
    all_res[index] = boxes

def adjust_box(x1, y1, x2, y2, w, h, k=5):
    x1 = max(x1, k)
    y1 = max(y1, k)
    x2 = min(x2, w-k)
    y2 = min(y2, h-k)
    return x1, y1, x2, y2

if __name__ == "__main__":
    dataset_name = "voc_2007_test"
    # for iter_num in [34999]:
    #     dir_name = "/gdata1/yangyc/wsddn/output/oicr_mt/vgg16_voc2007_oicr_mt_275_t4/"
    #     print(dir_name)
    #     detect_file = os.path.join(dir_name, 'detections.pkl')
    #     detect_nms_file = os.path.join(dir_name, 'detections_afternms.pkl')
    #     detect_bnms_file = os.path.join(dir_name, 'detections_beforenms.pkl')

    #     boxes, nms_boxes = boxes_nms(dataset_name, detect_file, detect_nms_file, detect_bnms_file)
    #     nms_boxes = filter_boxes(nms_boxes)
    #     sur_boxes = find_sur_boxes(nms_boxes, boxes)
    #     # sav_dir = "/ghome/yangyc/wsddn/outpic/oicr_mt/mt86_vis_test/%d/"%(iter_num)
    #     # boxes_draw_target(dataset_name, nms_boxes, draw_gt=True, sav_dir=sav_dir)
    #     sav_dir = "/gdata/yangyc/wsddn/outpic/oicr_mt/mt275_vis_test4_all2/%d/"%(iter_num)
    #     if not os.path.exists(sav_dir):
    #         os.makedirs(sav_dir)
    #     # boxes_draw_target(dataset_name, nms_boxes, draw_gt=True, sav_dir=sav_dir)
    #     # boxes_draw_target(dataset_name, nms_boxes, draw_gt=True, sav_dir=sav_dir, obj_num=0)
    #     boxes_draw_target(dataset_name, nms_boxes, draw_gt=False, draw_ok=False, sav_dir=sav_dir, same_color=True)
    #     # index_cls = ["bottle"]
    #     # # sav_dir = "/ghome/yangyc/wsddn/outpic/oicr_mt/mt86_vis_test/%d_neg/"%(iter_num)
    #     # sav_dir = "/gdata/yangyc/wsddn/outpic/oicr_mt/mt275_vis_test4/%d_neg_bottle/"%(iter_num)
    #     # boxes_draw_target(dataset_name, nms_boxes, draw_gt=True, draw_ok=False, index_cls=index_cls, sav_dir=sav_dir)

    for iter_num in [34999]:
        dir_name = "/gdata/yangyc/wsddn/output/vgg16_voc2007_oicr_bs3_3_1/test/model_step%d"%(iter_num)
        print(dir_name)
        detect_file = os.path.join(dir_name, 'detections.pkl')
        detect_nms_file = os.path.join(dir_name, 'detections_afternms.pkl')
        detect_bnms_file = os.path.join(dir_name, 'detections_beforenms.pkl')

        boxes, nms_boxes = boxes_nms(dataset_name, detect_file, detect_nms_file, detect_bnms_file, replace='/ghome/yinyf/wsddn/data/VOCdevkit/VOC2007/JPEGImages')
        nms_boxes = filter_boxes(nms_boxes)

        sav_dir = "/gdata/yangyc/wsddn/outpic/oicr_mt/bs3_3_1_all2/%d/"%(iter_num)
        boxes_draw_target(dataset_name, nms_boxes, draw_gt=False, draw_ok=False, sav_dir=sav_dir, same_color=True)