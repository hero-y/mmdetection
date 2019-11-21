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

_scale = 20
base_w = 1024
base_h = 1024
stride_w = 800
stride_h = 800
nms_thr = 0.5

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img_file', help='img file')
    parser.add_argument('--gpu', help='img file')
    args = parser.parse_args()
    return args


def main():
    #就按照1024,1024裁剪即可，最后输出的bbox的大小是乘以scale_factor的，也就是说是相对于1024来的，而不是800
    #不图片一共分成8个文件夹，分别用8个gpu
    args = parse_args()
    read = kfbReader.reader()
    model = init_detector(args.config, args.checkpoint, device=args.gpu) #'cuda:0' 
    if args.img_file is not None:
        test_names = os.listdir(args.img_file) #kfbs应该是训练数据集 path就是这些训练集的文件名 
        test_data_paths = [osp.join(args.img_file, test_name) for test_name in test_names]
        for test_data_path in test_data_paths: #对每一副图像处理
            name = osp.basename(test_data_path)
            print("name",name)
            save_path = ("/home/hero-y/data/cancer/crops/tianchi_cr18_a_d/"+ name).replace('.kfb','.json') 
            # print(save_path)
            save_data = []
            proposals_data = []
            read.ReadInfo(test_data_path,_scale,False)
            w = read.getWidth()
            h = read.getHeight()
            # print("w",w,"h",h)
            # print(int(h/base_h),int(w/base_w))
            for j in range(int(h/stride_h)):#对每个ROI遍历
                for i in range(int(w/stride_w)):
                    ROI = read.ReadRoi(i*stride_w, j*stride_h, base_w, base_h, _scale)           
                    result = inference_detector(model, ROI) #也是要经过pipeline的
                    bbox_result = result
                    bboxes = np.vstack(bbox_result) #pos是左上,右下的四个点(x,y,x,y)neg则是[],
                    if len(bboxes):#有值
                        for bbox in bboxes:
                            bbox[0] = (bbox[0] + i*stride_w)
                            bbox[1] = (bbox[1] + j*stride_h)
                            bbox[2] = (bbox[2] + i*stride_w)
                            bbox[3] = (bbox[3] + j*stride_h)
                            proposals_data.append(bbox.reshape(1,-1))
            proposals = np.concatenate(proposals_data,axis=0)
            print("proposals_pre",proposals.shape)
            proposals, _ = nms(proposals, nms_thr)
            print("proposals_pos",proposals.shape)
            for proposal in proposals:
                roi_save_data = {}
                roi_save_data["x"] = proposal[0].item() #必须要加.item()
                roi_save_data["y"] = proposal[1].item()
                roi_save_data["w"] = (proposal[2]-proposal[0]).item()
                roi_save_data["h"] = (proposal[3]-proposal[1]).item()
                roi_save_data["p"] = proposal[4].item()
                save_data.append(roi_save_data)
            with open(save_path, 'w') as f:
                json.dump(save_data, f)

            # proposals, _ = nms(proposals, cfg.nms_thr)  #nms使用的是ops/nms/nms_wrapper ,返回的是proposal的值，和有效的Proposal的序号
                    # print("i",i,"j",j)
                    # print(bboxes)
                    # print(save_data)
                    # show_result(ROI, result, model.CLASSES, score_thr=0.0)


# def main():
#     #就按照1024,1024裁剪即可，最后输出的bbox的大小是乘以scale_factor的，也就是说是相对于1024来的，而不是800
#     args = parse_args()
#     read = kfbReader.reader()
#     model = init_detector(args.config, args.checkpoint, device=args.gpu) #'cuda:0' 
#     if args.img_file is not None:
#         test_names = os.listdir(args.img_file) #kfbs应该是训练数据集 path就是这些训练集的文件名 
#         test_data_paths = [osp.join(args.img_file, test_name) for test_name in test_names]
#         for test_data_path in test_data_paths: #对每一副图像处理
#             # read.ReadInfo(test_data_path,_scale,False)
#             result = inference_detector(model, test_data_path) #也是要经过pipeline的
#             bbox_result = result
#             bboxes = np.vstack(bbox_result) #pos是左上,右下的四个点(x,y,x,y)neg则是[],
#             print(bboxes)
#             show_result(test_data_path, result, model.CLASSES, score_thr=0.05)


if __name__ == '__main__':
     main()
