import mmcv
import numpy as np
import torch

"""
共9个函数，分别对每个函数讲解，有几个还没用到
"""

"""
bbox2delta
在anchor_target和bbox_target中使用，首先求出proposal和gt_bbox的中线坐标和w,h,(p[...,2]+p[...,0])*0.5
在四个坐标里，0对应2,1对应3，w,h的时候要加上1
在求相对值，dx,dy,dw,dh;dx =(gx - px)/pw ,要减的是px,除的是pw,因为到时候还原的时候就是用p来和dx,dy,dw,dh操作的
torch.log(gw/pw),再用torch.stack拼接到一起
"""
def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw  #proposals,gt的含义:Proposals其实就是那个roi(在池化之前的)坐标，gt是这个proposal对应的ground_truth的坐标
    dy = (gy - py) / ph  #目的就是求出proposal和gt的中心坐标差，和wh之间的差距，用这些作为预测值
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas

#在rpn_proposal的时候，把根据cls_score挑出的2000个anchor当做proposal，根据他们对应的pred(delta)和对应的proposal,求出应当的bbox的位置
#也即是说，anchor的坐标是知道的，delta知道，那么就可以求出实际预测的bbox的位置了，而不是只用delta了
#生成的bbox才是真正的proposal，这些的坐标求出来后，就可以用nms了
"""
把roi或者anchor转换成算是预测的框的位置
先要把rois从(x,y,x,y)变成(x,y,w,h)才能和delta做操作，gw = pw*dw.exp(),gx = torch.addcmul(px,1,pw,dx)
接着在把(x,y,w,h)变回(x,y,x,y),此时左上角的x,y要各加0.5，右下角的xy要各减0.5，而在上面的bbox2delta中从x,y变w,h时，要各加1
"""
def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)  #.repeat(1,1)  deltas.size(1) // 4=1
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]  #双冒号代表 [开始：结尾：一步] range(10)[::3]是([0,3,6,9]),在这里[:,0::4]应该和[:,0]是一样的
    dy = denorm_deltas[:, 1::4] #(2000,1)
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip)) #4.135
    dw = dw.clamp(min=-max_ratio, max=max_ratio)  #是对数据做一下处理，gw / pw不能小于16/1000 ？？
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx #torch.addcmul(tensor, value=1, tensor1, tensor2)用tensor2对tensor1逐元素相乘，并对结果乘以标量值value然后加到tensor
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(Tensor or ndarray): Shape (..., 4*k)
        img_shape(tuple): Image shape.

    Returns:
        Same type as `bboxes`: Flipped bboxes.
    """
    if isinstance(bboxes, torch.Tensor):
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.clone()
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4] - 1
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4] - 1
        return flipped
    elif isinstance(bboxes, np.ndarray):
        return mmcv.bbox_flip(bboxes, img_shape)


def bbox_mapping(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * scale_factor
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape)
    return new_bboxes


def bbox_mapping_back(bboxes, img_shape, scale_factor, flip):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape) if flip else bboxes
    new_bboxes = new_bboxes / scale_factor
    return new_bboxes

"""
输入的是bbox_list,按图像分的，要达到的目标是变成(n,5),不但要把两个图像的bbox拼到一起，还要再加一个img_ind
先用torch.cat(,dim=-1),出来[batch_ind, x1, y1, x2, y2]，两个图像放到list中，再用一次torch.cat(dim=0)
"""
def bbox2roi(bbox_list): 
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2] 有n行，每行有5个值，分别是img_ind,和4个坐标
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list): #bbox_list的序号是图像的序号，迭代取出后，bboxes的shape是(n,4)
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id) #(n,1)数值全都是图像的id，即一个batch中的第几个图像
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)  #torch,cat([])参数是List，把list的矩阵，以特定维结合（把rois和图像id进行拼接）
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois) #把一个batch中的rois，放到一个List中
    rois = torch.cat(rois_list, 0) #把放了一个batch的rois的list，拼接(是把图像的rois之间进行拼接)
    return rois


def roi2bbox(rois):
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]#原始的bboxes和labels都是按照level进行cat的，现在变成按照类(label从0开始)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    return torch.stack([x1, y1, x2, y2], -1)
