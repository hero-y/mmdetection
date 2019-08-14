import torch

"""
一个类，有两个函数
初始化函数是为了获得三个部分的值：1.正样本和负样本的Bbox的坐标值2.正样本的对应的gt的Bbox的坐标值3.正样本的对应的gt的label值
bboxes函数就是把正样本和负样本的Bbox按顺序放到一起
"""
class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]  #把序号带入就有了坐标了
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1  #gt的序号带入gt_bbox的时候要减1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])  #注意这里bboxes,即采样的结果是把正样本和负样本分开放的，和anchor不同，anchor没有分开
