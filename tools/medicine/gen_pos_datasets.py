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
max_pos = 4 # 一个目标4次偏移

def inside(area, box):
    if box['x'] < area['x']:
        return False
    if box['y'] < area['y']:
        return False
    if box['x']+box['w'] > area['x']+area['w']:
        return False
    if box['y']+box['h'] > area['y']+area['h']:
        return False
    return True

def intersection(box, query_box):  #交并比
    portion = 0.
    iw = min(box[2], query_box[2]) - max(box[0], query_box[0]) + 1   
    if iw > 0:
        ih = min(box[3], query_box[3]) - max(box[1], query_box[1]) + 1
        if ih > 0:
            portion = iw * ih / ((query_box[2]-query_box[0]) * (query_box[3]-query_box[1]))
#            print(portion)
    return portion


def crop_samples(area, roi, name, data_root):
    W = roi[0]['w']
    H = roi[0]['h']
    # 切换坐标系并将字典转换成列表
    for i in range(1, len(roi)):
        box = roi[i]
        box['x'] = box['x'] - roi[0]['x'] #更新相对坐标系
        box['y'] = box['y'] - roi[0]['y']
        roi[i] = [box['x'], box['y'], box['x']+box['w'], box['y']+box['h']] #坐标变成左上右下
    roi.pop(0) #把第一个ROI剔除掉，剩下的都是bbox的坐标
    
    # 首先生成不包含target的crop
#     max_neg = 2 * len(roi)
#     neg_crops = []
#     n_neg = 0
#     temp = 0
#     max_temps = 100 * max_neg
#     while n_neg < max_neg:
#         temp += 1
#         if temp > max_temps:
#             print('Reach maximal temps')
#             break
#         crop_x = npr.randint(0, W-crop_size) #减crop_size的目的是如果起始坐标超过这个位置，就会超出边界
#         crop_y = npr.randint(0, H-crop_size)
#         crop_box = [crop_x, crop_y, crop_x+crop_size, crop_y+crop_size]
        
#         ovlp = False
#         for former_crop in neg_crops:
#             if intersection(former_crop, crop_box) > 0.2: #计算交并比把随机裁剪后的图片和之前的图比较，防止生成交并比过大的图
#                 ovlp = True
#                 break
#         if ovlp:
# #            print('ovlp')
#             continue
        
#         contain_target = False
#         for r in roi:
#             if intersection(crop_box, r) > 0.15: #防止里面有gt
#                 contain_target = True
#                 break
#         if contain_target:
# #            print('contain_target')
#             continue
        
#         neg_crops.append(crop_box)
#         n_neg += 1
    
#     for cnt, crop in enumerate(neg_crops):#每幅图中每个roi中的无gt的图片
#         neg_img = area[crop[1]:crop[3], crop[0]:crop[2]]
#         save_path = data_root + 'crops/notarget/' + name + '_neg-' + str(cnt+1) + '.png' #修改 TODO
#         cv2.imwrite(save_path, neg_img)
        
    """
    生成每个带gt的train
    """
    for i in range(len(roi)): #对每个图像的每个roi中的每个gt做循环
        x1 = max(roi[i][2] - crop_size, 0) #roi[i][2] - crop_size
        x2 = min(roi[i][0], W-crop_size)  #roi[i][0]
        y1 = max(roi[i][3] - crop_size, 0) #roi[i][3] - crop_size
        y2 = min(roi[i][1], H-crop_size)  #roi[i][1]
        
        n_pos = 0
        pos_crops = [[] for _ in range(max_pos)] # crop的坐标+内部target的坐标
        max_temps = 100 * max_pos
        temp = 0
        while n_pos < max_pos:  #max_pos=4
            temp += 1
            if temp > max_temps:
                print('pos: reach max')
                break
            crop_x = npr.randint(x1, x2)  #？？？
            crop_y = npr.randint(y1, y2)  #？？？
            crop_box = [crop_x, crop_y, crop_x+crop_size, crop_y+crop_size]
            # 保证对同一个target的crop重合度不是非常高
            very_close = False
            for cidx in range(n_pos):
                if intersection(pos_crops[cidx], crop_box) > 0.8:
                    very_close = True
            if very_close:
                continue
            # 符合要求，加入crop的坐标和target的坐标
            pos_crops[n_pos].extend(crop_box) #？？？
            pos_crops[n_pos].extend(roi[i]) #？？？
            # 检测其他target是否在该crop内
            for j in range(len(roi)):
                if j == i: continue
                if intersection(crop_box, roi[j]) > 0.5:
                    extra_target = [max(roi[j][0], crop_box[0]),
                                    max(roi[j][1], crop_box[1]),
                                    min(roi[j][2], crop_box[2]),
                                    min(roi[j][3], crop_box[3])]
                    pos_crops[n_pos].extend(extra_target)
            n_pos += 1
        
        for cnt, crop in enumerate(pos_crops): #因为每个bbox都生成了4幅图，这是对每幅图做的遍历
            if len(crop)==0: continue
            pos_img = area[crop[1]:crop[3], crop[0]:crop[2]]
            save_path = data_root + 'crops/train/' + name + '_pos-' + str(i+1) + '-' + str(cnt+1) + '.png'
            cv2.imwrite(save_path, pos_img)
            # 保存label
            save_path = save_path.replace('.png', '.json')
            save_path = save_path.replace('train', 'annotation_tmp')
            anno = []
            n_targets = int(len(crop) / 4 - 1)
            for k in range(1, n_targets+1):
                crop[4*k] -= crop[0]
                crop[4*k+1] -= crop[1]
                crop[4*k+2] -= crop[0]
                crop[4*k+3] -= crop[1]
                anno.append(crop[4*k: 4*k+4])
            #     cv2.rectangle(pos_img,(anno[k-1][0],anno[k-1][1]),(anno[k-1][2],anno[k-1][3]),(255,0,0))
            # cv2.imshow("pos_img",pos_img)
            # cv2.waitKey(0)
            with open(save_path, 'w') as f:
                json.dump(anno, f)
                
                    

if __name__ == '__main__':

    read = kfbReader.reader()

    data_root = '/media/hero-y/机械盘T1/Tianchi/medicine/'   
    train=dict(
        ann_file=data_root + 'annotations/',
        img_prefix=data_root + 'train/',
    )
    for pos_name in os.listdir(train['img_prefix']):
        print(pos_name)
        train_names = os.listdir(train['img_prefix']+pos_name)
        train_data_paths = [osp.join(train['img_prefix']+pos_name, train_name) for train_name in train_names]
        # print(train_data_paths)
    # train_names = os.listdir(train['img_prefix']) #kfbs应该是训练数据集 path就是这些训练集的文件名 
    # train_data_paths = [osp.join(train['img_prefix'], train_name) for train_name in train_names]
        for train_data_path in train_data_paths: #对每一副图像处理
            read.ReadInfo(train_data_path,_scale,False)
            # roi = read.ReadRoi(10240,10240,512,512,_scale)
            # print(read.getWidth(),read.getHeight())
            # cv2.imshow('roi',roi)
            # cv2.waitKey(0)
            label_path = train['ann_file'] + osp.basename(train_data_path).replace('kfb', 'json')
            with open(label_path ) as f:
                labels = json.load(f) #json.load相当于是把文件内容load进来
            rois = []
            poses = []
            for label in labels:
                if label['class'] == 'roi':
                    rois.append([label.copy()])
                elif label['class'] == 'pos':
                    poses.append([label.copy()])
            # print(rois)
            # print(poses)
            for roi in rois:
                for pos in poses:
                    if inside(roi[0], pos[0]): #
                        roi.append(pos[0].copy())
            # print(rois)
            name = osp.basename(train_data_path)[:-4] #每一副图像的名字
            cnt = 0
            for roi in rois: #对每一副图像中的每一个roi进行处理
                cnt += 1
                n = len(roi)
                label = roi[0]
                x = label['x']
                y = label['y']
                w = label['w']
                h = label['h']
                area = read.ReadRoi(label['x'], label['y'], label['w'], label['h'], _scale)
                crop_samples(area, roi, name+'_'+str(cnt), data_root)
        
        

    