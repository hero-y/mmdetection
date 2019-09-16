import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..registry import HEADS
from .guided_anchor_head import GuidedAnchorHead


@HEADS.register_module
class GARPNHead(GuidedAnchorHead):
    """Guided-Anchor-based RPN head."""

    def __init__(self, in_channels, **kwargs):
        super(GARPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        super(GARPNHead, self)._init_layers()

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        super(GARPNHead, self).init_weights()

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        (cls_score, bbox_pred, shape_pred,
         loc_pred) = super(GARPNHead, self).forward_single(x)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(GARPNHead, self).loss(
            cls_scores,
            bbox_preds,
            shape_preds,
            loc_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'],
            loss_rpn_bbox=losses['loss_bbox'],
            loss_anchor_shape=losses['loss_shape'],
            loss_anchor_loc=losses['loss_loc'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          mlvl_masks,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        #一次迭代的cls_scores:(1,h,w);bbox_preds:(4,h,w);mlvl_anchors：(k,4) 其中k=h*w ;mlvl_masks:(k,)
        #对于rpn中的预测值cls_scores,bbox_preds,shape_preds,loc_preds都是以level为list,每个的形状为(n,c,h,w)
        #而rpn中的生成的anchors都是以图像为list,每个level的形状为(k,4)
        #要按照先图像再level的形式对应起来，即如：bbox_preds[i][img_id],anchors[img_id][i]
        #再把shape对应起来，所以bbox_preds.permute(1,2,0).contiguous.view(-1,4)
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            anchors = mlvl_anchors[idx]
            mask = mlvl_masks[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1,
                                                                   4)[mask, :]
            if scores.dim() == 0:
                rpn_bbox_pred = rpn_bbox_pred.unsqueeze(0)
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # get proposals w.r.t. anchors and rpn_bbox_pred
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            #为什么要用9个anchor来确定这个位置的anchor应该对应哪个gt
            #因为该位置的anchor的大小是不确定的，所以就不能用它来选择对应的gt,
            #为什么不用guide_anchor来assign?
            #每个batch进来的时候都要重新生成anchor,因为图像的大小不一样
        
            # filter out too small bboxes
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            # NMS in current level
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            #对cat后的proposals(5个阶段在一起)进行nms,再切片取值
            #不用在multi levels中使用nms的原因是每个level是负责预测不同尺寸的物体的，如果对多个level之间使用nms
            #那么可能就会滤除掉一些在原本的level能够较好对应gt的proposals
            # NMS across multi levels
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            #进入 把各个阶段的proposals cat到一起后不用再nms，而是取scores的前topk的序号带入proposals中
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
