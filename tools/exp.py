import torch
import numpy as np
import cv2
from collections import defaultdict
import math

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
    print('val',val,'index',index)

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
    img = cv2.imread('../1.jpg')
    img2 = np.flip(img,axis=2)
    cv2.imshow('image',img2)
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

if __name__ == '__main__':
    



