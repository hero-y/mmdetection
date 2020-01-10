#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#偶数
#test_cancer_cr50_even中的保存路径要变,base_w等可能要变
#cfg文件要变
#模型文件要变
    
$PYTHON $(dirname "$0")/test_cancer_cr50_even.py \
    configs/cancer_final/cascade_rcnn_r50_fpn_1x_small800.py \
    work_dirs/cancer_final/cascade_rcnn_r50_fpn_1x_small800/epoch_12.pth \
    --img_file data/Data/test\
    --gpu 'cuda:0'
