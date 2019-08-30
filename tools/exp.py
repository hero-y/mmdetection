import torch
import numpy as np
import cv2
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import time
from random import  choice

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

def cuda_exp():
    """
    对于for i in range(m)，其中m来说就是普通的int,不是在cuda上
    使用.size(0),.shape[0]取出的也不是cuda
    对于在cuda的tensor上切片也可以不是cuda  n[1],n是cuda,1不是
    只有都在cuda上的tensor才能相加相与
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
    
if __name__ == '__main__':
