import torch

from .base_assigner import BaseAssigner
from .assign_result import AssignResult
from ..geometry import bbox_overlaps

"""
一个类MaxIoUAssigner,其中三个函数：__init__,assign,assign_wrt_overlaps(wrt:with regard to关于)
初始化就是把参数传入赋值给属性self.
assign里面主要是调用overlap,再调用assign_wrt_overlaps，返回的是AssignResult一个类

"""
class MaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    #用gt开头的变量,gt_bboxes,gt_labels里面的个数都是从json中获得的
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself  第四步是为了防止某些gt bbox没有匹配的proposal,因为如果两个gt离得很近，可能通过3步骤后，proposal找的都是另一个gt，就可能有gt被遗忘
        1.所有都置为-1  2.遍历每一个proposal和所有gt，若果都小于neg_thr则置为0  3.因为pos和neg阈值都是0.5，所以剩余的都是正的，把每个Proposal和最近的gt放到一起  4.为了防止某些gt没有pro所以有了第四步
        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]
        overlaps = bbox_overlaps(gt_bboxes, bboxes)  #把gt和每个bbox的iou都记录了下来

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0): #不进入
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1) 

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):  #判断变量是tuple还是float的好办法 ：isinstance
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0  #把proposal和gt的iou小于neg_iou_thr的proposal的标签当做是0
        #变成技巧很简单，没有遍历，而是先用max把每个proposal(anchor,在rpn中是anchor,在rcnn中是proposal)和gt的最大的求出来，再用<self.neg_iou_thr生成List True False,带入assigned_gt_ind即可      
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1  #赋的值不是1，而是该幅图中的第几个gt

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1  #这个if的意思是：或许一个gt会存在有多个proposal的iou的值是最大的(相等的)，这个if是把这些proposal都设置Ind
                else:                                       #+1是因为gt的排序默认是从0开始的，所以要+1，让它从1开始
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1 #这里面就直接用了gt_argmax_overlaps，因为max函数当里面的iter有多个值是最大且相等时只会把最后一个的ind取出来：max(1,2,2),只会取最后一个2

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()  #这句话的目的是为了把Proposal中的pos(对应gt大于0)的序号取出来
            #torch.nonzero：返回的是(z,n)z是输入值中大于零的数字的个数，n是输入的维度，里面的数值是符合的数的坐标
            #例子：torch.nonzero([1,2,-1,3,0]>0) 其实就是torch.nonzero(1,1,0,1,0),一共有3个大于0，该维度是1所以输出的是(3,1),再用.squeeze就变成(3,)
            #输出的是序号:(0,1,3)
            if pos_inds.numel() > 0:  #.numel()是计算元组里面值的数量，len是求list的
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]  #pos_inds是对应gt大于0的序号，把这些proposal的label设置成gt_label,assigned_gt_inds[pos_inds],是对应的gt的序号
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
