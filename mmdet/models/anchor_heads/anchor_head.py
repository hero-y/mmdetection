from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from ..builder import build_loss
from ..registry import HEADS

"""
一个AnchorHead类，继承了nn.Module,其中一共有10个函数
分别是一个初始化函数，一个初始化层函数，和一个初始化权重函数
forward和forward_single函数，loss和loss_single函数，get_anchors和get_bboxes和get_bboxes_single函数
初始化函数：传值，构造loss函数赋值给self.,定义每个层的AnchorGenerator放到List中，这其中生成了基本anchor
初始化层函数就是定义cls和bbox的卷积函数，输入值是输入通道，输出通道，核，赋值给属性
初始化权重函数没用
forward函数就是把图像输入到初始化层的那两个属性中
loss函数，调用了grid_anchors,在每一层上生成了anhcor,此时还是按图像排列的，每个图像的list中有5个层，这里的每个层的anchor已经是K*A了，即将来loss的时候就不用像cls_score和bbox_pred一样要用permute转换维度(把通道维放到最后)
又调用了anchor_target,生成的标签值就已经是按照层级排列了，不再是按图像排列的
调用loss_single,输入的cls_score,bbox_pred和那些标签值都是按层级排列的，最后是把每一层的loss都求了，在解析loss时，才求的均值
loss_single中，对输入的值都做了reshape，也就是相当于把一个层的这些预测值或者标签值都整合到了一起，注意的是cls_score和bbox_pred用了permute再reshape的
get_bboxes是用来生成bbox_head用的proposal的，是对一个一个图像的pred和anchor单独操作的，先生成每个层级的anchor放到mlvl_anchors的list中
在这里的序号是没有图像的，再用for in 对图像循环，因为cls_score和bbox_pred都是按照层级分的，所以这个时候要把图像和图像的层级区分出来，
[cls_score[i][img_id] for i in range(num_levels)],再调用rpn中的get_bboxes_single生成proposals，该proposals是一个图像的，把它append到result_list

对于anchor来说：先生成基本anchor,再生成全部anhcor再assign_and_sample,再生成target，所有的anchor都有target,是因为定义的时候全都默认为0，就只把sample后的pos(label把neg也算上了)赋予真正的target
assign_and_sample和生成target都在anchor_target中
对于proposal来说：也是assign_and_sample和生成target，不过bbox_targets中只有生成target，assign_and_sample分开的
"""
@HEADS.register_module
class AnchorHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False) #dict中的get，获取某一个键值的值，如果不存在，则赋值为后面的那个参数
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1  #num_classes=2,判断是物体还是背景，-1后就输出了一个概率，所以proposal是5个值
        else:
            self.cls_out_channels = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.fp16_enabled = False

        self.anchor_generators = []  #定义一个List，对anchor的基本大小做迭代，实例化了每一层的AnchorGenerator,都append到这个List中
        if isinstance (self.anchor_scales[0],list):
            for i ,anchor_base in enumerate(self.anchor_base_sizes): #用for in做了一次迭代，每次只输入一个anchor_base进去
                anchor_scales = self.anchor_scales[i]
                self.anchor_generators.append(
                    AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
            self.num_anchors = len(self.anchor_ratios) * len(anchor_scales)
        else:
            for anchor_base in self.anchor_base_sizes: #用for in做了一次迭代，每次只输入一个anchor_base进去
                self.anchor_generators.append(
                    AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
            self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

    def forward_single(self, x):
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas): #通过调用anchor_generators中的函数构建list
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time 因为是一个Batch里面的
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        #每个图像都有multi_level_anchors，对于同一个Batch的图像而言，他们的w,h是相同的，这样才能算一个batch,所以他们的anchor也是相同的
        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []  #存的是不同图片的
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = [] #存的是一张图片的不同阶段的valid_flag
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)  #np.ceil是向上取整，但其实h/anchor_stride=feat_h
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list #二者都是里面有多个图像，每个图像有多个层级

    def loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)  #label是(2,K*A) K*A的排序方式是：把一个位置的3个anchor排一起，依次排其余位置，从左到右，从上到下
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)  #经实验验证，这样做确实和label一样，把一个位置的3个anchor排一起，依次排其余位置
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        #featmap.size() torch.Size([2, 3, 56, 80])  featmap.size()[-2:] featmap.size()[-2:] (n,c,h,w)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] #把cls_scores的5个特征图的wh取出来
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets  #这里的list都是按照层级排序的，而cls_scores和bbox_preds也是按照层级list的，便于Loss
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox = multi_apply(  #在这里的参数cls_scores到bbox_weights_list都是List,对象是层级，就一个一个对象的输入，所以输出的也是个list
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)  #losses_cls和losses_bbox是两个List,里面包含的是5个层级的Loss,最后再parser_loss函数(api/train.py)中才把这5个相加
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self, cls_scores, bbox_preds, img_metas, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]  #根据每个level的特征图的大小和stride在原图上生成anchor,每个对象是(k×A，4)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  #.detach()是阻断反向传播,detach()用于返回一个新的从当前的Variable图中分离的Variable，返回的Variable不需要梯度
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            #self.get_bboxes_single调用的是rpn_head中的，不是下面的
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        #cls_score一次迭代的shape:(1*3,h,w),bbox_pred:(4*3,h,w),anchors:(K*3,4),k=h*w
        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            #.permute(1,2,0)的目的是把通道维数放到最后的位置，所以就是最先被取出来的，所以把一个位置的anchor放到一起
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()  #二分类问题，使用sigmoid将pred限制在(0,1)
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            #5个层每个层都根据cls_score取前2000个anchor,如果高层的anchor数没那么多，就全都用了
            if nms_pre > 0 and scores.shape[0] > nms_pre:  #scores.shape[0]所有的anchor数
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)  #其实是把每个anchor的score都取出了(k*3,)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]  #类别分数前2000个anchor (2000,4)
                bbox_pred = bbox_pred[topk_inds, :] #(2000,4)
                scores = scores[topk_inds, :] #(2000,)
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)  #
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)  #(mlvl_scores.shape[0],2) (一共的proposal的数量,2)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
