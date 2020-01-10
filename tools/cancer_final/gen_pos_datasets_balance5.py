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
#crop_size = 1024
crop_size = 1280 #1536/64=24
crop_shape = (crop_size,crop_size)
max_pos = 1 # 一个目标4次偏移

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

def intersection(box, query_box_input):  #交并比,在这里的交并比实际上是相交的面积/gt的面积，而不是传统的两个面积之和
    portion = 0.
    query_box = [query_box_input['x1'],query_box_input['y1'], query_box_input['x2'], query_box_input['y2']]
    iw = min(box[2], query_box[2]) - max(box[0], query_box[0]) + 1   
    if iw > 0:
        ih = min(box[3], query_box[3]) - max(box[1], query_box[1]) + 1
        if ih > 0:
            portion = iw * ih / ((query_box[2]-query_box[0]) * (query_box[3]-query_box[1]))
#            print(portion)
    return portion

def intersection_for_crops(box_input, query_box):  #交并比
    portion = 0.
    box = [box_input[0][0],box_input[0][1],box_input[0][2],box_input[0][3]]
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
    # 切换坐标系
    for i in range(1, len(roi)):
        box = roi[i].copy()
        box['x'] = box['x'] - roi[0]['x'] #更新相对坐标系
        box['y'] = box['y'] - roi[0]['y']
        # roi[i] = [box['x'], box['y'], box['x']+box['w'], box['y']+box['h']] #坐标变成左上右下
        del roi[i]['x']
        del roi[i]['y']
        del roi[i]['w']
        del roi[i]['h']
        roi[i]['x1'] = box['x']
        roi[i]['y1'] = box['y']
        roi[i]['x2'] = box['x']+box['w']
        roi[i]['y2'] = box['y']+box['h']
    roi.pop(0) #把第一个ROI剔除掉，剩下的都是bbox的坐标
    # print("roi_new",roi)
    #到这里之前每个roi都转成了相对坐标，每个roi都是个List,其中有多个dict,每个dict里面是每个gt的内容:
    #形如[{'class': 'ASC-H', 'x1': 7567, 'y1': 2175, 'x2': 7710, 'y2': 2329}, {'class': 'HSIL', 'x1': 1231, 'y1': 3393, 'x2': 1432, 'y2': 3552}]
        
    """
    生成每个带gt的train
    最后裁剪好的每个图片的json文件格式为:[{'x1':,'y1':,'x2':,'y2':,'class':},{}...]
    """
    a = b = 0
    for i in range(len(roi)): #对每个图像的每个roi中的每个gt做循环
        "x1,x2指的是左上角x点的变化区间，y1,y2指的是左上角y点的变化区间"
        x1 = max(roi[i]['x2'] - crop_size, 0) #roi[i][2] - crop_size
        x2 = min(roi[i]['x1']+1, W-crop_size+1)  #roi[i][0]
        y1 = max(roi[i]['y2'] - crop_size, 0) #roi[i][3] - crop_size
        y2 = min(roi[i]['y1']+1, H-crop_size+1)  #roi[i][1]

        #对于x1,x2来说，如果W-crop_size都<0,那么roi[i]['x2']一定小于零
        if x2<=x1 and x2<=0: #针对小图做处理 设置标志位？
            x2 = roi[i]['x1']+1
        if x2<=x1 and x2>0  :#针对超级大图的处理
            a = x1
            x1 = x2
            x2 = a
        if y2<=y1 and y2<=0:##针对小图做处理 设置标志位？
            y2 = roi[i]['y1']+1
        if y2<=y1 and y2>0  :#针对超级大图的处理
            b = y1
            y1 = y2
            y2 = b

        #'ASC-H','ASC-US','HSIL','LSIL','Candida'
        if roi[i]['class'] == 'ASC-H':
            max_pos = 1
        if roi[i]['class'] == 'ASC-US':
            max_pos = 1
        if roi[i]['class'] == 'HSIL':
            max_pos = 2
        if roi[i]['class'] == 'LSIL':
            max_pos = 2
        if roi[i]['class'] == 'Candida':
            max_pos = 5

        n_pos = 0
        pos_crops = [[] for _ in range(max_pos)] # crop的坐标+内部target的坐标
        max_temps = 200 * max_pos
        temp = 0
        while n_pos < max_pos:  #max_pos=4
            temp += 1
            if temp > max_temps:
                print('pos: reach max')
                break
            crop_x = npr.randint(x1, x2+2)  
            crop_y = npr.randint(y1, y2+2)  
            crop_box = [crop_x, crop_y, crop_x+crop_size, crop_y+crop_size]
            # 保证对同一个target的crop重合度不是非常高,也就是保证对同一个gt生成的4个裁剪区域，不会高度重合
            very_close = False
            for cidx in range(n_pos):
                if intersection_for_crops(pos_crops[cidx], crop_box) > 0.8:
                    very_close = True
            if very_close:
                continue
            # 符合要求，加入crop的坐标和target的坐标
            pos_crops[n_pos].append(crop_box) #pos_crops是一个List,其中有4个位置，每个位置是针对每个gt生成的坐标，其中包含图坐标以及gt坐标
            pos_crops[n_pos].append(roi[i]) 
            #pos_crops整体形如:[[],[],[],[]],其中每个[]是[[],{},{}..],第一个[]代表裁剪的区域的坐标，后面的{}是其中的每个gt的坐标
            #注意此时{}内的坐标还没有转换成相对裁剪区域的坐标
            # 检测其他target是否在该crop内
            for j in range(len(roi)):
                if j == i: continue
                if intersection(crop_box, roi[j]) > 0.5:
                    extra_target = {}
                    extra_target['x1'] = max(roi[j]['x1'], crop_box[0])
                    extra_target['y1'] = max(roi[j]['y1'], crop_box[1])
                    extra_target['x2'] = min(roi[j]['x2'], crop_box[2])
                    extra_target['y2'] = min(roi[j]['y2'], crop_box[3])
                    extra_target['class'] = roi[j]['class']
                    pos_crops[n_pos].append(extra_target)
            n_pos += 1
        # print("pos_crops",pos_crops)
        # print(len(pos_crops))
        
        for cnt, crop in enumerate(pos_crops): #因为每个bbox都生成了4幅图，这是对每幅图做的遍历
            if len(crop)==0: continue
            # print("crop_all",crop)
            pos_img = area[crop[0][1]:crop[0][3], crop[0][0]:crop[0][2]]#裁图是先y再x

            crop_shape = (crop_size,crop_size) + (pos_img.shape[-1], )
            pad = np.empty(crop_shape, dtype=pos_img.dtype)
            pad[...] = 0
            pad[:pos_img.shape[0], :pos_img.shape[1], ...] = pos_img
            pos_img = pad

            save_path = data_root + 'crops_balance5/train/' + name + '_pos-' + str(i+1) + '-' + str(cnt+1) + '.png'
            cv2.imwrite(save_path, pos_img)
            # 保存label
            save_path = save_path.replace('.png', '.json')
            save_path = save_path.replace('train', 'annotation_tmp')
            anno = []
            n_targets = int(len(crop)-1)

            for k in range(1, n_targets+1):
                label = {}
                label['x1'] = max(crop[k]['x1'] - crop[0][0], 0)
                label['y1'] = max(crop[k]['y1'] - crop[0][1], 0)
                label['x2'] = min(crop[k]['x2'] - crop[0][0], crop_size)
                label['y2'] = min(crop[k]['y2'] - crop[0][1], crop_size)
                label['class'] = crop[k]['class']
                anno.append(label)
            #     cv2.rectangle(pos_img,(anno[k-1]['x1'],anno[k-1]['y1']),(anno[k-1]['x2'],anno[k-1]['y2']),(255,0,0))
            # cv2.imshow("pos_img",pos_img)
            # cv2.waitKey(0)
            with open(save_path, 'w') as f:
                json.dump(anno, f)
                
                    

if __name__ == '__main__':

    read = kfbReader.reader()

    # data_root = '/home/admin/jupyter/train_data_split/' 
    # crop_data_root = '/home/admin/jupyter/' 
    data_root = '/home/hero-y/train_data_split/' 
    crop_data_root = '/home/hero-y/' 
    train=dict(
        ann_file=data_root + 'annotations/',
        img_prefix=data_root + 'pos/',
    )
    pos_cnt = 0
    for pos_name in os.listdir(train['img_prefix']):
        pos_cnt += 1
        print("pos_cnt",pos_cnt)
        print(pos_name)
        train_data_path = osp.join(train['img_prefix'],pos_name)
        read.ReadInfo(train_data_path,_scale,False)
        label_path = train['ann_file'] + osp.basename(train_data_path).replace('kfb', 'json')
        with open(label_path ) as f:
            labels = json.load(f) #json.load相当于是把文件内容load进来
        rois = []
        poses = []
        for label in labels:
            if label['class'] == 'roi':
                rois.append([label.copy()])#要加[]的原因是下面要对每个roi进行append，dict是没办法append的
            elif label['class'] in ['ASC-H','ASC-US','HSIL','LSIL','Candida']:
                poses.append([label.copy()])

        if len(poses) != 0:#防止一个ann文件中只存在Ts那个类的情况
            for roi in rois:
                for pos in poses:
                    if inside(roi[0], pos[0]): 
                        roi.append(pos[0].copy())
        else:
            continue

        # print("roi",roi)
        #到这个地方为止，每个roi都是一个list,其中是很多个dict
        name = osp.basename(train_data_path)[:-4] #每一副图像的名字
        cnt = 0
        for roi in rois: #对每一副图像中的每一个roi进行处理
            cnt += 1
            n = len(roi)
            label = roi[0]
            area = read.ReadRoi(label['x'], label['y'], label['w'], label['h'], _scale)
            crop_samples(area, roi, name+'_'+str(cnt), crop_data_root)
            #图片命名原则:图+第几个roi+pos+第几个gt+gt中的第几个
        

    