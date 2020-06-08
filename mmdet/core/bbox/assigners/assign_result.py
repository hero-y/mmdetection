import torch

"""
AssignResult中只有一个类，有__init__和add_gt_
gt_inds,max_overlaps,labels的个数都是proposal的个数
__init__主要目的是取出现在的Proposal对应的gt序号，和最大的overlap和对应的labels
再用torch.cat把这三项的gt加上分别是：torch.cat([torch.arange(1,len(gt_labels)+1,device='cuda'),self.gt_inds])
torch.cat([self.max_overlaps.new_ones(num_gts),self.max_overlaps])
torch.cat([gt_labels,self.labels])
"""
class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds #每一个proposal对应的gt的序号
        self.max_overlaps = max_overlaps #每一个proposal对应的最大的iou
        self.labels = labels #每一个proposal对应的gt的label

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
