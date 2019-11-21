#coding=utf-8
import kfbReader
import cv2
import os
import os.path as osp
import sys
import json
import numpy as np
import numpy.random as npr

npr.seed(213)
_scale = 20
crop_size = 1024
stride_h = 1024
stride_w = 1024
base_w = 1024
base_h = 1024

if __name__ == '__main__':

    read = kfbReader.reader()
    data_root = '/media/hero-y/机械盘T1/Tianchi/medicine/'
    train=dict(
        img_prefix=data_root + 'neg/',
    )
    for neg_name in os.listdir(train['img_prefix']): #对每一个neg文件夹操作
        print(neg_name)
        train_names = os.listdir(train['img_prefix']+neg_name)
        train_data_paths = [osp.join(train['img_prefix']+neg_name, train_name) for train_name in train_names]
        for train_data_path in train_data_paths: #对每一副图像处理
            img_name = osp.basename(train_data_path)[:-4] #每一副图像的名字
            print(img_name)
            read.ReadInfo(train_data_path,_scale,False)
            w = read.getWidth()
            h = read.getHeight()
            for j in range(int(h/stride_h)):#对每个ROI遍历
                for i in range(int(w/stride_w)):
                    save_path = data_root + 'crops/neg_datasets/' + img_name + '_neg-' + str(i+1) + '-' + str(j+1) + '.png' 
                    neg_img = read.ReadRoi(i*stride_w, j*stride_h, base_w, base_h, _scale)
                    cv2.imwrite(save_path, neg_img)
                    # 保存label
                    save_path = save_path.replace('.png', '.json')
                    save_path = save_path.replace('neg_datasets', 'annotation_tmp_neg')
                    anno = []
                    anno.append([0,0,0,0])
                    with open(save_path, 'w') as f:
                        json.dump(anno, f)
        #             break
        #         break
        #     break
        # break