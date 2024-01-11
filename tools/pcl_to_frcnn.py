"""Perform re-evaluation on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time
from six.moves import cPickle as pickle
import numpy as np

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import empty_results, extend_results
from core.test import box_results_for_corloc, box_results_with_nms_and_limit
from utils.io import save_object
from datasets.json_dataset import JsonDataset
from datasets import task_evaluation
import utils.logging
import utils.boxes as box_utils

# from tqdm import tqdm

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--result_path',
        help='the path for result file.')
    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results.')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')
    parser.add_argument(
        '--style', 
        help='map or corloc',
        default='map')
    return parser.parse_args()


if __name__ == '__main__':

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert os.path.exists(args.result_path)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.result_path)
        logger.info('Automatically set output directory to %s', args.output_dir)

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2014":
        cfg.TEST.DATASETS = ('coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 80
    elif args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 80
    elif args.dataset == 'voc2007test':
        cfg.TEST.DATASETS = ('voc_2007_test',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012test':
        cfg.TEST.DATASETS = ('voc_2012_test',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2007trainval':
        cfg.TEST.DATASETS = ('voc_2007_trainval',)
        cfg.MODEL.NUM_CLASSES = 20
    elif args.dataset == 'voc2012trainval':
        cfg.TEST.DATASETS = ('voc_2012_trainval',)
        cfg.MODEL.NUM_CLASSES = 20
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Re-evaluation with config:')
    logger.info(pprint.pformat(cfg))

    with open(args.result_path, 'rb') as f:
        results = pickle.load(f)
        logger.info('Loading results from {}.'.format(args.result_path))
    all_boxes = results['all_boxes']
    # print(list(all_boxes.keys())[:10])
    # input()

    dataset_name = cfg.TEST.DATASETS[0]
    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt=True)
    final_boxes = []
    valid_pic = 0
    TYPES = [1, 2]
    # TYPES = [0]
    # vis = True
    vis = False
    print(cfg.TEST.SCORE_THRESH)
    if cfg.OICR.Need_Reg:
        print("Need_Reg")
    # print(list(all_boxes.keys())[0])
    # for i, entry in tqdm(enumerate(roidb)):
    for i, entry in enumerate(roidb):
        gt_classes = entry['gt_classes'][0]
        boxes_img = all_boxes[entry['image'].replace('pcl', 'wsddn')]
        scores, boxes = boxes_img['scores'], boxes_img['boxes']
        
        num_classes = cfg.MODEL.NUM_CLASSES + 1
        cls_boxes = [[] for _ in range(num_classes)]

        bbox_list = []
        for j in range(1, num_classes):
            if gt_classes[j-1] == 1:
                # apply nms
                inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
                scores_j = scores[inds, j]
                if cfg.OICR.Need_Reg:
                    boxes_j = boxes[inds, j * 4:(j + 1) * 4]
                else:
                    boxes_j = boxes[inds, :]
                dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(np.float32, copy=False)
                keep = box_utils.nms(dets_j, cfg.TEST.NMS)
                nms_dets = dets_j[keep, :]
                cls_boxes[j] = nms_dets
                # apply thershold
                
                nms_dets_t = nms_dets[nms_dets[:, -1] > 0.4]
                for box_ind in nms_dets_t:
                    bbox_list.append({
                        "boxes": box_ind[:4],
                        "cls": j-1,
                        "score": box_ind[-1]
                    })
                if 1 in TYPES:
                    if len(nms_dets_t) == 0:
                        ind = np.argmax(nms_dets[:, -1])
                        nms_dets = nms_dets[ind]
                        if nms_dets[-1] >= 0.15 and (2 in TYPES):
                            bbox_list.append({
                                    "boxes": nms_dets[:4],
                                    "cls": j-1,
                                    "score": nms_dets[-1]
                                })
                
                # print(nms_dets)
                # print(max(nms_dets[:, -1]))
                

        final_boxes.append({
            "id" : entry['image'].split('/')[-1].split('.')[0],
            "bbox" : bbox_list
        })
        if len(bbox_list) > 0:
            valid_pic += 1

        if vis:
            im = cv2.imread(entry['image'])
            sav_img_name = entry['image'].split('/')[-1]
            yaml_name = args.cfg_file.split('/')[-1].split('.')[0]
            output_pic_dir = "/ghome/yinyf/wsddn/outpic/momentum_test/{}".format(yaml_name)
            if not os.path.exists(output_pic_dir):
                os.makedirs(output_pic_dir)
                
            for proposal in bbox_list:
                bbox = proposal['boxes']
                score = proposal['score']
                bbox = tuple(int(np.round(x)) for x in bbox[:4])
                cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 0, 204), 2)       # red
                cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
            sav_pic = os.path.join(output_pic_dir, sav_img_name)
            cv2.imwrite(sav_pic, im)

    print("valid_pic: ", valid_pic)
    res_file = os.path.join(args.output_dir, 'oicr_mm43_iter2_a2_final_04_015.pkl')
    print(res_file)
    save_object(final_boxes, res_file)