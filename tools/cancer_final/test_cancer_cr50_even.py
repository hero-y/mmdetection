#coding=utf-8
from mmdet.apis import init_detector, inference_detector, show_result, show_result_pyplot
import mmcv
import argparse
import time
import kfbReader
import numpy as np
import os
import os.path as osp
import json
from mmdet.ops import nms
import torch

_scale = 20
base_w = 800 #1280
base_h = 800
stride_w = 400 #640
stride_h = 400
nms_thr = 0.5

class_names = ['ASC-H', 'ASC-US', 'HSIL', 'LSIL', 'Candida', 'Trichomonas']

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_file', help='img file')
    parser.add_argument('--gpu', help='gpu')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    read = kfbReader.reader()
    model = init_detector(args.config, args.checkpoint, device=args.gpu) #'cuda:0'
    
    test_data_names = []
    test_json_names = [] 
    cnt = 0
    print(os.path.abspath(args.img_file ))
    if args.img_file is not None:
        test_names = os.listdir(args.img_file) #kfbs应该是训练数据集 path就是这些训练集的文件名 
        for test_name in test_names:
            if test_name.endswith(".kfb"):
                test_data_names.append(test_name) 
            if test_name.endswith(".json"):
                test_json_names.append(test_name) 

        test_data_paths = [osp.join(args.img_file, test_name) for test_name in test_data_names]
        test_json_paths = [osp.join(args.img_file, test_name) for test_name in test_json_names]

        for test_data_path in test_data_paths: #对每一副图像处理
            name = osp.basename(test_data_path)
#             name_prefix = int(name[:4])
#             if name_prefix%2 == 1:
#                 continue
            cnt += 1
            print("name",name,"cnt",cnt)
            test_json_path = osp.join(args.img_file, name).replace('.kfb','.json') 
            save_path = ("/home/admin/jupyter/results/test_cancer_cr50_small800_1024/"+ name).replace('.kfb','.json') 
#             save_path = ("/home/hero-y/results/test_cancer_cr50/"+ name).replace('.kfb','.json') 
            with open(test_json_path,'r') as f:
                roi_labels = json.load(f)
            save_data = []
            read.ReadInfo(test_data_path,_scale,False) 
            for roi_label in roi_labels:
                area = read.ReadRoi(roi_label['x'], roi_label['y'], roi_label['w'], roi_label['h'], _scale)
                # print(save_path)
                w = roi_label['w']
                h = roi_label['h']
                # print("w",w,"h",h)
                # print(int(h/base_h),int(w/base_w))
                proposals_data = []
                labels_data = []
                Candida_data = []
                Candida_data_label = []
                for j in range(int(h/stride_h)+1):#对每个ROI遍历
                    for i in range(int(w/stride_w)+1):
                        detect_img = area[j*stride_h:(j*stride_h+base_h),i*stride_w:(i*stride_w+base_w)]

                        crop_shape = (base_h, base_w) + (detect_img.shape[-1], )
                        pad = np.empty(crop_shape, dtype=detect_img.dtype)
                        pad[...] = 0
                        pad[:detect_img.shape[0], :detect_img.shape[1], ...] = detect_img
                        detect_img = pad

                        result = inference_detector(model, detect_img) #也是要经过pipeline的
                        bbox_result = result
                        bboxes = np.vstack(bbox_result) #pos是左上,右下的四个点(x,y,x,y)neg则是[],

                        labels = [
                                np.full(bbox.shape[0], i, dtype=np.int32)
                                for i, bbox in enumerate(bbox_result)
                            ] #这种形式对每个class进行循环,bbox.shape[0]代表的是该类中bbox的个数,可能为0
                        labels = np.concatenate(labels)#用concatenate后就会把空的去掉 
                        #到这里，bboxes是(n,5);labels是(n,)
                        if len(bboxes):#有值
                            for bbox in bboxes:
                                bbox[0] = (bbox[0] + i*stride_w + roi_label['x'])
                                bbox[1] = (bbox[1] + j*stride_h + roi_label['y'])
                                bbox[2] = (bbox[2] + i*stride_w + roi_label['x'])
                                bbox[3] = (bbox[3] + j*stride_h + roi_label['y'])                            
                                proposals_data.append(bbox.reshape(1,-1))
                        labels_data.append(labels)
        
                if len(labels_data)==0 or len(proposals_data)==0:
                    continue
                labels_all = np.concatenate(labels_data,axis=0)
                proposals = np.concatenate(proposals_data,axis=0)
                print("proposals_pre",proposals.shape)

                ###对一个图片经过重叠预测后，应该一类一类的nms
                AH_inds = np.nonzero(labels_all == 0)
                AS_inds = np.nonzero(labels_all == 1)
                HL_inds = np.nonzero(labels_all == 2)
                LL_inds = np.nonzero(labels_all == 3)
                CA_inds = np.nonzero(labels_all == 4)
                TS_inds = np.nonzero(labels_all == 5)

                AH_proposals = proposals[AH_inds]
                AS_proposals = proposals[AS_inds]
                HL_proposals = proposals[HL_inds]
                LL_proposals = proposals[LL_inds]
                CA_proposals = proposals[CA_inds]
                TS_proposals = proposals[TS_inds]

                if len(AH_proposals) != 0:
                    AH_proposals, _ = nms(AH_proposals, nms_thr)
                if len(AS_proposals) != 0:
                    AS_proposals, _ = nms(AS_proposals, nms_thr)
                if len(HL_proposals) != 0:
                    HL_proposals, _ = nms(HL_proposals, nms_thr)
                if len(LL_proposals) != 0:
                    LL_proposals, _ = nms(LL_proposals, nms_thr)
                if len(CA_proposals) != 0:
                    CA_proposals, _ = nms(CA_proposals, nms_thr)
                if len(TS_proposals) != 0:
                    TS_proposals, _ = nms(TS_proposals, nms_thr)
                
                AH_labels = np.ones((len(AH_proposals),), dtype=np.int32)*0
                AS_labels = np.ones((len(AS_proposals),), dtype=np.int32)*1
                HL_labels = np.ones((len(HL_proposals),), dtype=np.int32)*2
                LL_labels = np.ones((len(LL_proposals),), dtype=np.int32)*3
                CA_labels = np.ones((len(CA_proposals),), dtype=np.int32)*4
                TS_labels = np.ones((len(TS_proposals),), dtype=np.int32)*5

                proposals = np.concatenate((AH_proposals,AS_proposals,HL_proposals,LL_proposals,CA_proposals,TS_proposals),axis=0)
                labels_all = np.concatenate((AH_labels,AS_labels,HL_labels,LL_labels,CA_labels,TS_labels),axis = 0)


#                 proposals, inds = nms(proposals, nms_thr)#对全图进行nms
#                 labels_all = labels_all[inds]
                print("proposals_pos",proposals.shape)
                print("labels_all",labels_all.shape)
                for proposal,label_all in zip(proposals,labels_all):
                    roi_save_data = {}
                    roi_save_data["x"] = proposal[0].item() #必须要加.item()
                    roi_save_data["y"] = proposal[1].item()
                    roi_save_data["w"] = (proposal[2]-proposal[0]).item()
                    roi_save_data["h"] = (proposal[3]-proposal[1]).item()
                    roi_save_data["p"] = proposal[4].item()
                    roi_save_data["class"] = class_names[label_all]
                    save_data.append(roi_save_data)
            with open(save_path, 'w') as f:
                json.dump(save_data, f)

if __name__ == '__main__':
     main()
