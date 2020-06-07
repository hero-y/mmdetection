import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES
from .utils import weight_reduce_loss

"""
一个类CrossEntropyLoss,继承nn.Module,里面有两个函数初始化和forward,注意用forward是因为要backward,
否则直接用__call__也可以调用，外面有三个函数
初始化函数是把三个loss函数作为属性传了进来，并判断要用哪个loss函数，forward就调用loss函数，并加了loss_weight
"""
#对于anchor来说输入的pred和label是所有anchor的，并且也都计算了loss，但是因为有label_weights,
#label_weights会对pos和neg都计算，而bbox_weights只有pos才是1
#anchor中的cls输出通道只有一个值(因为sigmoid)，p是判断是物体的概率值，通过cross_entropy，如果是物体(label=1)则-log(p),此时p越大代表越可能是物体，loss就越小，如果不是(label=0)则-log(1-p),函数中自动实现
#proposal中的cls会输出所有类的概率值，此时就只是对应的那个位置的-log(p)
def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    loss = F.cross_entropy(pred, label, reduction='none')  #交叉熵损失函数的pred是(N,C),lable是(N)
    # 也就是说label就是每个Roi就一个标签，pred是每个roi的所有类都有概率值，最后交叉熵实现的也就是：对应的那个类的概率的Log的负数，因为log的输入值是小于1的，所以负负得正
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor) #avg_factor的值是N，实现的就是loss.sum() / avg_factor
    return loss


def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

"""
mask分支使用的loss函数，输入的是通过网络输出的每个pro的pred和每个pro的target，以及每个pro的label，有label是因为每一个类都有pred,所以要取出label对应的
pred和target的size()[0]都是一个batch中所有图像的pro的和
首先获得正pro的个数.size()[0],用arange(0,num_rois)做出一个序列，把该序列和label同时带入pred对pred进行切片
为了是对每个proposal按顺序操作，取出每个pro的pred对应label位置的预测
"""
def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    # TODO: handle these two reserved arguments
    #pred:(n,81,28,28) target:(n,28,28) label:(n,) 注意这里通道数是81，论文中也说了实例分割任务中对每一类单独使用sigmoid效果好
    #所以使用pred[inds, label]把对应位置的序号取出来，然后使用binary_cross_entropy_with_logits(专门用于sigmoid的函数)
    #并也可以说明gt_mask的值只有0,1,然后通过label来进行限定具体
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)#(n,28,28) 
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, reduction='mean')[None]#二进制交叉熵函数，是因为target的值是0或1，with_logits是内部自带了sigmoid,公式是-[t*log(p)+(1-t)*log(1-p)]


@LOSSES.register_module
class CrossEntropyLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy  #sigmoid把函数映射到(0,1)之间
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls
