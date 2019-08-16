import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss

"""
一个类SmoothL1Loss,类中初始化函数和forward函数，类外smooth_l1_loss
smooth_l1_loss先用abs求差的绝对值，再用torch.where作为判断语句
"""
@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):  #rpn的beta是1.0/9.0
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)  #torch.where()就是一个判断语句,beta是系数，然后再用装饰器，去做Loss平均值
    #0.5x*x    (|x|<1.0)
    #|x|-0.5   (|x|>1.0)
    #注意这里的Loss是(n,4)形状的，pred中的4个值不是x,y,wh,而是转化过得，4个值每一个都有Loss值，最后求了平均
    return loss


@LOSSES.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
