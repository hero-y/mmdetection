
#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON $(dirname "$0")/test_cancer_all_cr18.py \
    configs/cancer/cascade_rcnn_r18_fpn_20_a_d.py \
    work_dirs/cancer/cascade_rcnn_r18_fpn_20e/epoch_20.pth \
    --img_file ../data/cancer/crops/test/test_all \
    --gpu 'cuda:0'