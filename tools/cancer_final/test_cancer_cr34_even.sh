#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#偶数

# $PYTHON $(dirname "$0")/test_cancer_cr34_even.py \
#     configs/cancer_final/cascade_rcnn_r34_fpn_1x.py \
#     work_dirs/cancer_final/cascade_rcnn_r34_fpn_1x/epoch_12.pth \
#     --img_file ../data/Data/test\
#     --gpu 'cuda:0'

$PYTHON $(dirname "$0")/test_cancer_cr34_even.py \
    configs/cancer_final/cascade_rcnn_r34_fpn_1x.py \
    work_dirs/cancer/cascade_rcnn_r34_fpn_20e/epoch_20_2.pth \
    --img_file ../data/Data/test\
    --gpu 'cuda:0'