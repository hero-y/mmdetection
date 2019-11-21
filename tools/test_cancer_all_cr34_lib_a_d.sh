
#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON $(dirname "$0")/test_cancer_all_cr34_lib_a_d.py \
    configs/cancer/libra_cascade_rcnn_r34_fpn_2x_a_d.py \
    work_dirs/cancer/libra_cascade_rcnn_r34_fpn_2x_a_d/epoch_24.pth \
    --img_file ../data/cancer/crops/test/test_all \
    --gpu 'cuda:0'