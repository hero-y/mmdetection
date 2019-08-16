import torch

from ..utils import multi_apply
from .transforms import bbox2delta

"""
在bbox_target.py文件中有三个函数
bbox_target主要就是调用了multi_apply,和concat把两幅图像的数据合并到一个tensor里面 torch.cat
bbox_target_single中没有assign_sample过程，对于bbox来说该过程早就完成了
bbox_target_single主要就是建立变量labels,label_weights,bbox_targets,bbox_weights
对应序号赋值就可，bbox_targets的赋值是bbox2delta后的
第三个函数expand_target还没用到
"""
def bbox_target(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_bboxes_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_bboxes_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:  #把两个图像中的值拼接到一起
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_bboxes,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels  #bbox_target中label中传入的是pos的数量即可，因为采样后的bbox按照正负样本排列，所以pred也是按照正负样本排的，所以label也要这样
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox2delta(pos_bboxes, pos_gt_bboxes, target_means,
                                      target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0  #在分类任务中，正负样本都用了，因为label_weight的正负样本的权重都为1，而在bbox的坐标任务中，只用了正样本的Loss,控制用没用是通过权重来控制的，bbox_weights的负样本的权重为0

    return labels, label_weights, bbox_targets, bbox_weights


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros(
        (bbox_targets.size(0), 4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros(
        (bbox_weights.size(0), 4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
