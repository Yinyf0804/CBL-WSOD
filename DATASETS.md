We follow the instructions from [pcl.pytorch-0.4.0](https://github.com/ppengtang/pcl.pytorch/tree/0.4.0) to prepare datasets.

### Prepare Dataset (VOC07&12)
1. Download the training, validation, test data and VOCdevkit
  ```Shell
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
  ```
2. Extract all of these tars into one directory named `VOCdevkit`

  ```Shell
  tar xvf VOCtrainval_06-Nov-2007.tar
  tar xvf VOCtest_06-Nov-2007.tar
  tar xvf VOCdevkit_18-May-2011.tar
  ```
3. Download the COCO format pascal annotations from [here](https://drive.google.com/drive/folders/1R4leOIYxP9qHJ2dVQJ4fKv2CoEHeEu41?usp=sharing) and put them into the `VOC2007/annotations` directory

4. It should have this basic structure

  ```Shell
  $VOC2007/                           
  $VOC2007/annotations
  $VOC2007/JPEGImages
  $VOC2007/VOCdevkit        
  # ... and several other directories ...
  ```

4. Create symlinks for the PASCAL VOC dataset

  ```Shell
  cd $FIWSOD_ROOT/data
  ln -s $VOC2007 VOC2007
  ```
5. [Optional] follow similar steps to get PASCAL VOC 2012.

### Prepare Proposals (Selective Search)
1. The Selective Search proposals can be downloaded [here](https://drive.google.com/drive/folders/1dAH1oPZHKGWowOFVewblSQDJzKobTR5A?usp=sharing). You should put the generated proposal data under the folder $FIWSOD_ROOT/data/selective_search_data:
  ```Shell
  $FIWSOD_ROOT/data/selective_search_data
  $FIWSOD_ROOT/data/selective_search_data/voc_2007_trainval.pkl
  $FIWSOD_ROOT/data/selective_search_data/voc_2007_test.pkl
  $FIWSOD_ROOT/data/selective_search_data/voc_2012_trainval.pkl
  $FIWSOD_ROOT/data/selective_search_data/voc_2012_test.pkl
  ```