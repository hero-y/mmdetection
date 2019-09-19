import torch
import numpy as np
import cv2
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import time
from random import  choice
from torch.autograd import Variable

def exp_dim():
    m = torch.randn(2, 1, 4)
    n = torch.randn(1, 2)
    o = torch.randn(2, 4)
    a = torch.randn(1, 2)
    b = torch.randn(1, 2)
    c = torch.cat((a, b), dim=-1)
    d = torch.stack((a, b, n), dim=-1)
    e = torch.argmax(m, dim=-1)
    f = m.transpose(0,1)
    g = m.permute(1,0,2)
    h = m.squeeze() #squeeze里面有参数，就代表只看该维是否是1
    i = m.unsqueeze(2)
    val,index = o.max(dim=1) #和topk一样显示val，再是Index
    sort_val, indices = torch.sort(o, dim=1, descending = True) #dim=1，代表列发生变化

    na = np.arange(12).reshape(3,4)
    nb = np.arange(12).reshape(3,4)
    print(np.vstack((na,nb)).shape) #(6,4)((na,nb))里面要打括号，因为这样子就相当于是一个参数的位置了
    print(np.hstack((na,nb)).shape) #(3,8)
    print(np.concatenate((na,nb),axis=1).shape) #(3,8)
    print('a', a)
    print('b', b)
    print('n', n)
    print('c', c.shape)
    print('d', d.shape,d)
    print('e', e.shape)
    print('f', f.shape)
    print('g', g.shape)
    print('h', h.shape)
    print('i', i.shape)
    print('o',o)
    print('val',val,'index',index)
    print('sort_val',sort_val,'\n','indices',indices)

def print_indx():
    print('loss[{0}]'.format(1))
    print('loss[%s]'%(1))

def diff_size_shape():
    """
    a.size是返回a中元素的个数，a.size(0)是返回第0维的元素的个数，所以用()
    a.shape返回的是a的形状，a.shape[0]返回的是a的形状中的第一个的数字大小
    """
    a=torch.randn(3,2)
    print(a.size)
    print(a.size(1))
    print(a.shape)
    print(a.shape[1])

def exp_reshape():
    """
    reshape的操作，是从左到右，从上向下去reshape的
    shape的结构对应于特征图的理解:如下(2,3,4,5),想象特征图的样子，第一行是[0,1,2,3,4],第二行是[5,6,7,8,9]
    依次类推，对于第一个面加上一个括号[]，一共三个维，每个维也都加上一个[]，最后整个一张特征图加上一个[],剩下的那个特征图同理，从左到右，从上向下
    """
    target =torch.arange(0,120).reshape(2,3,4,5)
    print('target',target)
    print('target',target.reshape(-1,1))  #reshape时从左到右，从上向下的去排


def coco_exp():
    """
    self.anns = anns #输入ann的Id获取该ann的字段
    self.imgToAnns = imgToAnns #输入img的id，获取该ann的字段
    self.catToImgs = catToImgs #输入类别的id获取该图像的字段
    self.imgs = imgs #输入图像的id获取图像的字段
    self.cats = cats #输入类别的id获取类别的字段
    """
    ann = {'Y':1,'N':0}
    imgToAnns = defaultdict(list)
    imgToAnns[10].append(ann) #当键值为数字的时候，不用引号，此时的键值为10，因为imgToAnns设置为dictlist，所以赋值的时候用的是append
    imgToAnns[10].append(ann)
    print(imgToAnns[10],ann['Y'])

    left = np.array(5)  #在mmdet.datasets.extra_aug中
    top = np.array(6)
    print(np.tile((left, top), 2))
    b = np.array([[1, 2], [3, 4]])
    print(np.tile(b, 2)) #沿X轴复制2倍
    print(np.tile(b, (2, 1)))#沿X轴复制1倍（相当于没有复制），再沿Y轴复制2倍

    a = np.arange(10).reshape(2,5)
    print(a[:,None].shape, a[:,:,None].shape)  #a[:,None]因为按顺序，第一个是:,那None就是第二个，即使没有写a[:,None,:]
    print(a[...,None].shape)  #把前面的:都省去，None在最后一维


def cv2_exp():
    """
    在图像上画框，并在左上角涂满颜色，并在中间写上类别
    注意：更改了mmcv中的Image.py文件，使得test-show的时候也有这个效果，该文件可以通过
    mmdet/models/detectors/base.py最下面的mmcv.imshow_det_bboxes找到
    在if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
    和cv2.putText中间添加了如下四行
    size = cv2.getTextSize(label_text,cv2.FONT_HERSHEY_COMPLEX,font_scale,1)
    text_width = size[0][0]
    text_height = size[0][1]
    cv2.rectangle(
        img, (bbox_int[0],bbox_int[1]-text_height-2),(bbox_int[0]+text_width,bbox_int[1]), bbox_color, thickness=-1)
    在上面的基础上添加代码使得每个类别画出的框的颜色不同：
    在text_color = color_val(text_color)和for bbox, label in zip(bboxes, labels)之间添加：
    colors = ['red','green','blue','cyan','yellow','magenta']
    old_label = 0
    在for bbox, label in zip(bboxes, labels)和bbox_int = bbox.astype(np.int32)之间添加:
    即：
    for bbox, label in zip(bboxes, labels):#开始
        if old_label != label:
            if len(colors) == 0 :
                colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']
            bbox_color = choice(colors)
            colors.remove(bbox_color)
            bbox_color = color_val(bbox_color)
        old_label = label
        bbox_int = bbox.astype(np.int32)#截止
    并在开头添加from random import  choice
    """
    img = cv2.imread('../1.jpg')
    img2 = np.flip(img,axis=2) #随机翻转
    bbox = np.array((30,50,200,300))
    bbox_int = bbox.astype(np.int32)
    left_top = (bbox_int[0],bbox_int[1])
    right_bottom = (bbox_int[2],bbox_int[3])
    cv2.rectangle(
        img, left_top, right_bottom, (0, 0, 255), thickness=2)
    label_text = 'horse'
    label_text += '|{:.02f}'.format(0.7)
    size = cv2.getTextSize(label_text,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
    text_width = size[0][0]
    text_height = size[0][1]
    cv2.rectangle(
        img, (bbox_int[0],bbox_int[1]-text_height-2),(bbox_int[0]+text_width,bbox_int[1]), (0, 0, 255), thickness=-1)
    cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
    cv2.imshow('image',img)
    cv2.waitKey (0)
    cv2.destroyAllWindows()

    """
    rgb的图像和bgr的图像的shape都是(h,w,3),对于rgb来说通道3的顺序就是rgb,对于bgr来说通道3的顺序是bgr
    当使用的是caffe的预训练模型时,to_rgb=False,因为caffe出现的比较早,兼容了opencv用的是bgr,而对于torch来说to_rgb=True
    当to_rgb=False时：mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0]
    当to_rgb=True时：mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    可以看出对于mean来说只是把第一个和第三个位置换了一下,数值大小差不多,std3个之间差不多
    img_norm_cfg的数值应该就是使用cv2.meanStdDev求出每个图像的mean和std，再求平均
    """
    img = cv2.imread('../1.jpg')
    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2默认为bgr顺序
    img_rgb_mean, img_rgb_std = cv2.meanStdDev(img_rgb)
    img_rgb_mean = img_rgb_mean.squeeze(1)
    img_rgb_std = img_rgb_std.squeeze(1)
    img_rgb = (img_rgb-img_rgb_mean)/img_rgb_std
    cv2.imshow('img_rgb', img_rgb)
    cv2.waitKey (0)
    cv2.destroyAllWindows()


def cuda_exp():
    """
    对于for i in range(m)，其中m来说就是普通的int,不是在cuda上
    使用.size(0),.shape[0]取出的也不是cuda
    对于在cuda的tensor上切片也可以不是cuda  n[1],n是cuda,1不是
    只有都在cuda上的tensor才能相加相与
    """
    """
    detach()讲解：
    .detach()是阻断反向传播,detach()用于返回一个新的从当前的Variable图中分离的Variable，返回的Variable不需要梯度
    Variable是torch.autograd中的一个包，当Variable(tensor)输入的tensor的变化过程就会被Variable记录下来
    需要在anchor_head中get_bboxes中的cls_score和bbox_pred中使用，这样是因为cls_score和bbox_pred
    不但要用来计算loss也需要用来生成接下来的proposal，当生成proposal的时候是不需要计算梯度的，所以用.detach(),生成一个
    不需要计算梯度的cls_score和bbox_pred
    同样的在guided_anchor_head中get_guided_anchors_single中在生成guided_anchor时，loc_pred和shape_preds也要detach()
    在get_bboxes中的cls_score和bbox_pred和guide_anchors和loc_mask需要detach()
    """
    m = torch.tensor([0,2,2,1,1], device='cuda')
    print(m>1) #tensor([0, 1, 1, 0, 0], device='cuda:0', dtype=torch.uint8)
    #从cpu变到cuda,cpu上的数据是不能和gpu的数据想乘相加的,必须都在gpu上
    #1.使用torch.tensor(,device='cuda') 2.torch.tensor().cuda() 3.torch.tensor().to(Device)
    #4.先定义在cpu上，再to(Device)，但要注意此时那个数据还在cpu上，只有赋值后的新数据才在gpu上
    Device = torch.device('cuda:0')
    a = torch.tensor([1,2,3],device='cuda:0')
    b = torch.tensor([
        [2], [3],[4]
    ])
    c = b.cuda() #如果在定义b的时候就.cuda()那么b就在cuda上，而现在b是在cpu上，c是在cuda上
    print(a*c)

def index_exp():
    m = [0,2]
    rois = torch.randn(3,4)
    index = rois.new_zeros((3,1))
    index[2,:] = 1
    rois = torch.cat([index,rois],dim=1)
    print("rois原",rois)
    #rois[:0] == 0,通过>,<,==等生成一个长度和原长度一致的0,1序列，在多维的时候常用来判断某一列的值，就使用[:,i]的方法
    inds = torch.nonzero(rois[:,0] == 0).squeeze() #torch.nonzero把非零位置的索引值取出来，inds是个tensor的list
    #rois[:0] == 0 是tensor([1,1,0]),inds是tensor([0,1])
    b1 = rois[inds,1:]
    print("rois经过一个tensor的List作为索引值之后",b1)
    b2 = rois[m,1:] #对于一个tensor可以把一个List作为索引值来切片，该list可以是tensor，也可以不是
    print("rois经过一个普通的List作为索引值之后",b2)
    inds2 = torch.ones(len(rois),dtype = torch.uint8)
    m = torch.tensor([0,1])
    inds2[:len(m)] = m
    b3 = rois[inds2,1:] #也就是说用切片的方法对一个tensor取出一部分，可以是把对应的索引的位置带入，也可以是把一串[0,1]带入，这串值的大小和总索引的长度一样
    print("rois经过一个和他的长度一样有0,1组成的list作为索引值之后",b3)
    m = np.array([[0, 0, 1, 0],
                  [0, 1, 1, 0]])
    if (m > 0).any:#.any或.all用来判断某个array是否有True或都是True
        print(m[m > 0])#切片操作

    """
    seq[start:end:step]
    """
    n = range(100)[3:18:2]

    """
    使用None的方法扩展维度，inside_flags[:,None]是(3,1);.expand相当于是复制了，inside_flags[:,None].expand(-1,9)是(3,9)
    """
    inside_flags = torch.tensor([3,4,5])
    expand_inside_flags = inside_flags[:,None].expand(-1,9).reshape(-1)
    print(expand_inside_flags)


def plt_exp():
    """
    画出RetinaNet论文中的图1，即参数r对loss的影响
    """
    r = np.array([0, 0.5, 1, 2, 5])
    x = np.linspace(0,1)

    y0 = np.power((1 - x), r[0])
    y0 = -y0 * np.log(x)
    y1 = np.power((1 - x), r[1])
    y1 = -y1 * np.log(x)
    y2 = np.power((1 - x), r[2])
    y2 = -y2 * np.log(x)
    y3 = np.power((1 - x), r[3])
    y3 = -y3 * np.log(x)
    y4 = np.power((1 - x), r[4])
    y4 = -y4 * np.log(x)

    plt.xlabel("probablity of ground truth class")#x轴上的名字
    plt.ylabel("loss")#y轴上的名字

    plt.plot(x, y0, 'b', label='r = 0')
    plt.plot(x, y1, 'r', label='r = 1')
    plt.plot(x, y2, 'y', label='r = 2')
    plt.plot(x, y3, 'pink', label='r = 3')
    plt.plot(x, y4, 'g', label='r = 4')
    plt.legend(loc = 'upper right')
    plt.show()

    """
    画smooth_l1_loss的beta取值不同时的图
    """
    beta1 = 1.0
    beta2 = 0.11
    diff = torch.linspace(0,2,steps=50)
    loss1 = torch.where(diff < beta1, 0.5 * diff * diff / beta1,
                       diff - 0.5 * beta1)
    loss2 = torch.where(diff < beta2, 0.5 * diff * diff / beta2,
                    diff - 0.5 * beta2)
    diff = diff.numpy()
    loss1 = loss1.numpy()
    loss2 = loss2.numpy()
    plt.plot(diff,loss1,'r',diff,loss2,'b')
    plt.show()

    """
    画出各阈值下的AP图(cascade rcnn中的图5),iou=0.9时AP=-1,比较奇怪舍去
    运行test_by_result文件,输入config restult eval三个参数(result形如.pkl.bbox.json)便可以通过
    之前输出的结果文件进行coco评估
    """
    x = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.95])#IOU阈值
    y = np.array([0.585,0.567,0.545,0.517,0.484,0.439,0.379,0.300,0.038])#AP
    plt.xlim((0.5, 0.95))
    plt.ylim((0, 0.7))
    plt.plot(x,y)
    plt.show()

    """
    在cascade rcnn中的图4(The IoU histogram of training samples)的画法
    hist输入的是一个数据组,用hist的方法可以自动计算出某个区间内的数据的数量,而bins代表是框数
    在画某个iou区间pro的数量的时候,只需要把所有的pro的iou放到一个numpy中,再调用hist即可
    """
    data = np.random.rand(2000)
    data = data[data >= 0.5]
    plt.hist(data, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlim((0.5, 1.0))
    plt.xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()

def basic_grammer_exp():
    """
    torch中定义变量时dtype的使用；torch.floor是向下取整,torch.ceil是向上取整；.clamp是限制大小；.item()是取出元素值
    """
    scale = torch.tensor([[54],[122]],dtype=torch.float)#在定义变量时,标注dtype,如dtype=torch.float
    min_anchor_size = torch.tensor([32],dtype=torch.float)
    target_lvls = torch.floor(
        torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)#torch.floor向下取整
    target_lvls = target_lvls.clamp(min=0, max=4).long()
    print(target_lvls[0].item())

    """
    如cfg中的model是一个dict,dict比较适合在初始化时传入参数，实际是调用了build_from_cfg，
    在这里面当把type pop出来之后，输入的时候其实是**args
    """
    """
    扩展维度的方法:expand_as,expand,repeat,
    expand_as:输入的是某一个变量，最后输出的值的shape和输入的变量的shape一致
    expand:最后的输出的shape就是epxand中传入的两个值
    repeat:输入的第一个参数代表沿着行重复的次数，第二个参数代表沿着列重复的次数,此时是把输入就当做一个元素进行重复
    """
    points = torch.randn(12,2)
    expanded_regress_ranges = points.new_tensor((-1,64))[None].expand_as(points)

    num_points = 10
    areas = torch.tensor([1,2,3,4,5])
    areas1 = areas[None].repeat(num_points, 1)#repeat的第一个参数代表沿着行重复的次数，纵坐标代表沿着列重复的次数,此时是把输入就当做一个元素进行重复的
    areas2 = areas[None].expand(num_points,areas.size(0))#expand,最后的输出的shape就是epxand中传入的两个值

    """
    gt_labels是(2,4)可以一次性输入多个索引值，如min_area_inds是三个索引值
    """
    gt_labels = torch.arange(8).reshape(2, 4)
    min_area_inds = torch.tensor([1, 1, 0])
    labels = gt_labels[min_area_inds]

    """
    bbox_targets是在fcos中的，(3,2,4)意思是3个点,每个点中有2个gt预选,每个gt的坐标是4个值
    为了求出每个点中合适的gt的坐标值，有如下三种写法
    bbox_targets1相当于是对每个点中去3次情况的gt(bbox_targets1是错误的写法)
    bbox_targets2和bbox_targets3效果一样,bbox_targets3更容易理解(二者都是正确的写法)
    """
    bbox_targets = torch.arange(24).reshape(3, 2, 4)
    bbox_targets1 = bbox_targets[:, min_area_inds]  # (3,3,4)
    bbox_targets2 = bbox_targets[range(bbox_targets.size(0)), min_area_inds]
    bbox_targets3 = []
    for i in range(bbox_targets.size(0)):
        bbox_targets3.append(bbox_targets[i, min_area_inds[i]])
    bbox_targets3 = torch.cat(bbox_targets3, dim=0).reshape(bbox_targets.size(0), 4)

    """
    按照指定的维度根据规定的大小num_points,对tensor进行分割,分割出一个个新的tensor,这些tensor在一个tuple中
    """
    labels = torch.randn(12)
    num_points = [3, 5, 4]
    labels_list = labels.split(num_points, 0)

    """
    nonzero返回一个包含输入中所含元素非零索引的张量,输出张量中每行包含输入中非零元素的索引,所以是(n,1),故一般nonzero后要加上.reshape(-1)
    """
    pos_inds = torch.tensor([0, 1, 1, 0, 1]).nonzero().reshape(-1)

    """
    可以再tensor中使用if i in tensor: 去判断某个元素是否在该tensor中
    """
    labels = torch.tensor([0, 1])
    if 0 in labels:
        print(0)
if __name__ == '__main__':
