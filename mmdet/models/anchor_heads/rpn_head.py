import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from .anchor_head import AnchorHead
from ..registry import HEADS

"""
一个RPNHead类，继承AnchorHead,一共有6个函数
分别是初始化函数，初始化层函数，初始化权重函数
forward_single函数，loss函数，get_bboxes_single函数
初始化层函数多了一个rpn_conv
get_bboxes_singles是一层一层的提取proposals,这里就是一幅图像，在anchor_head的get_bboxes中有对图像的循环，也就是说大循环是对图像，内部有多层级的小循环

"""
@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox']) #losses是一个dict，rpn的loss返回的是dict(loss_cls=,loss_bbox=),在该loss中就调用了这个键值,losses['loss_cls'],loss['loss_bbox']

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx] #用序号取出特定层的值
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0) #和anchor_head中loss_single的permute目的一样。都是对预测值进行permute,变成通道的三个值在一起，才能reshape
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid() #使用了.sigmoid
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)  #没用sigmoid话，列维度就是2
                scores = rpn_cls_score.softmax(dim=1)[:, 1]  #使用softmx,按照维度，把数据缩放到(0,1)区间，且该维度的数据之和为1，之后再取出第二维的数据
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre: #分数的个数比nms前的阈值要大，在浅层会进入，高层就不会
                _, topk_inds = scores.topk(cfg.nms_pre)  #根据score的大小只要取出序号就可以，带入rpn_bbox_pred，anchors，scores
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:  #求出每个proposals的w,h然后和cfg.min_bbox_size做比较，再用&的方式去求出valid_flag
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)  #每个proposal的维度变成5,总为(n,5),前四个是坐标，最后一个是scores
            #设置score:1.用来在anchore中根据score来选择前2000个anchor(每个层级的前2000个) 2.用来做nms
            proposals, _ = nms(proposals, cfg.nms_thr)  #nms使用的是ops/nms/nms_wrapper ,返回的是proposal的值，和有效的Proposal的序号
            proposals = proposals[:cfg.nms_post, :]  #cfg.nms_post是2000，代表nms后最对每层有多少个proposal，这是因为如果nms_pre设置的很大如4000个nms后也很多，此时nms_post才会起作用，但目前nms_pre和nms_post都是2000，则nms_post就没什么作用了
            mlvl_proposals.append(proposals) #.append是在for中使用的，每结束一个就添加一个
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else: #进入
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])  #nms_pre和nms_post都是代表每个level的最多的Proposal的数量，而max_num代表所有层级加到一起的数量
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
