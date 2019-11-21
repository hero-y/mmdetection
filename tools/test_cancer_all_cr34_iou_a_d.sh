
#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON $(dirname "$0")/test_cancer_all_cr34_iou_a_d.py \
    configs/cancer/cascade_rcnn_r34_fpn_20e_a_d.py \
    work_dirs/cancer/cascade_rcnn_r34_fpn_20e/epoch_20_2.pth \
    --img_file ../data/cancer/crops/test/test_all \
    --gpu 'cuda:0'