import _init_paths
import pickle
import copy
import numpy as np
from pycocotools.coco import COCO
from datasets.dataset_catalog import ANN_FN
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import IM_DIR
from datasets.dataset_catalog import IM_PREFIX

class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name, proposal_file, output_file, name_ori):
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.COCO = COCO(DATASETS[name][ANN_FN])

        self.name_ori = name_ori
        self.image_directory_ori = DATASETS[name_ori][IM_DIR]
        self.COCO_ori = COCO(DATASETS[name_ori][ANN_FN])

        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.proposal_file = proposal_file
        self.output_file = output_file
        # self.get_roidb()
        self._get_gt()

    
    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        # Include proposals from a file
        self._add_proposals_from_file(
            roidb, self.proposal_file, min_proposal_size, proposal_limit,
            crowd_filter_thresh
        )
        return roidb

    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        with open(proposal_file, 'rb') as f:
            proposals = pickle.load(f)
        proposals_ori = proposals.copy()
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        ori_item = len(proposals['boxes'])
        box_list = []
        id_lists = [entry['id'] for entry in roidb]
        proposal_ind_list = [i for i, index in enumerate(proposals[id_field]) if index in id_lists]
        print(len(id_lists), len(proposal_ind_list))
        if 'scores' in proposals:
            fields_to_sort = ['boxes', id_field, 'scores']
        else:
            fields_to_sort = ['boxes', id_field]
        for k in fields_to_sort:
            proposals[k] = [proposals[k][i] for i in proposal_ind_list]
        
        for i, entry in enumerate(roidb):
            boxes = proposals['boxes'][i]
            ind_ori = proposals_ori[id_field].index(entry['id'])
            print(boxes)
            print(proposals_ori['boxes'][ind_ori])
            assert str(entry['id']) == str(proposals[id_field][i])
            input()

        with open(self.output_file, 'wb') as f:
            pickle.dump(proposals, f)
        print('Saved in {} for {} --> {} items'.format(self.output_file, ori_item, len(proposals['boxes'])))
        
    def _get_gt(self):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))

        image_ids_ori = self.COCO_ori.getImgIds()
        image_ids_ori.sort()
        roidb_ori = copy.deepcopy(self.COCO_ori.loadImgs(image_ids))
        id_lists_ori = [entry['id'] for entry in roidb_ori]

        for i, entry in enumerate(roidb):
            ind_ori = id_lists_ori.index(entry['id'])
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            for obj in objs:
                cls = self.json_category_id_to_contiguous_id[obj['category_id']]
                print('cur', cls)
            
            entry_ori = roidb_ori[ind_ori]
            ann_ids_ori = self.COCO.getAnnIds(imgIds=entry_ori['id'], iscrowd=None)
            objs_ori = self.COCO.loadAnns(ann_ids_ori)
            for obj_ori in objs_ori:
                cls_ori = self.json_category_id_to_contiguous_id[obj_ori['category_id']]
                print('ori', cls_ori)

            input()

def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    if 'scores' in proposals:
        fields_to_sort = ['boxes', id_field, 'scores']
    else:
        fields_to_sort = ['boxes', id_field]
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]

if __name__ == "__main__":
    # name = 'voc_2012_train'
    # proposal_file = '/ghome/yinyf/pcl/data/selective_search_data/voc_2012_trainval.pkl'
    # output_file = '/ghome/yinyf/pcl/data/selective_search_data/voc_2012_train.pkl'
    name = 'voc_2012_val'
    proposal_file = '/ghome/yinyf/pcl/data/selective_search_data/voc_2012_trainval.pkl'
    output_file = '/ghome/yinyf/pcl/data/selective_search_data/voc_2012_val.pkl'
    name_ori = 'voc_2012_trainval'
    JsonDataset(name, proposal_file, output_file, name_ori)