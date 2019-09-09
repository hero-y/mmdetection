from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_inside_flags, anchor_target,
                        delta2bbox, force_fp32, ga_loc_target, ga_shape_target,
                        multi_apply, multiclass_nms)
from mmdet.ops import DeformConv, MaskedConv2d
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from .anchor_head import AnchorHead

"""
提出DCNV1是因为在传统conv中输出特征图的某个(x,y)位置的感受野形状是固定的，四四方方的，所以用了offset后，该位置的感受野
可以相当于是有一些特殊形状的。而在guided_anchor中对特征图使用，是因为当anchor的形状不固定后，原位置(x,y)的感受野可能就
不能满足这种形状的anchor的需求，所以用deform_conv后就能够满足了

类FeatureAdaption，其中有初始化函数来传入参数，并构建所要用的网络；有初始化权重函数；有forward函数
DCNv1由两个部分组成，在论文中是把输入特征图通过1*1卷积生成offset图，再和原输入特征图一同输入DeformableConv中
在这里是把预测的shape(2,2,h,w)输入1*1的conv中，输出的shape是(2,72,h,w),72=3*3*2*4,输出通道之所以这样设置
是因为输出的那个位置(x,y)是由kernal为(3,3)即原图的9个元素分别乘w相加得到的，此时要对这3*3个元素做offset，而偏移的方向是2D的
所以输出是3*3*2，而后面的4代表deformable_groups,也就相当于是Mobilenet中的group吧，在这里使用是因为输出的某个位置(x,y)的通道数是256
可能不想让这256个数都是用一种offset得到的，设置group为4，就相当于是有4组offset,每个group使用一组offset
"""
class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2 
        self.conv_offset = nn.Conv2d(
            2, deformable_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())#先使用conv_offset输入shape_pred预测偏移量，每次输入的shape是一个level的(2,2,h,w)
        x = self.relu(self.conv_adaption(x, offset))
        return x

"""
相比AnchorHead多了get_sampled_approxs函数，loss_shape_single，loss_loc_single
"""
@HEADS.register_module
class GuidedAnchorHead(AnchorHead):
    """Guided-Anchor-based head (GA-RPN, GA-RetinaNet, etc.).

    This GuidedAnchorHead will predict high-quality feature guided
    anchors and locations where anchors will be kept in inference.
    There are mainly 3 categories of bounding-boxes.
    - Sampled (9) pairs for target assignment. (approxes)
    - The square boxes where the predicted anchors are based on.
        (squares)
    - Guided anchors.
    Please refer to https://arxiv.org/abs/1901.03278 for more details.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        octave_base_scale (int): Base octave scale of each level of
            feature map.
        scales_per_octave (int): Number of octave scales in each level of
            feature map
        octave_ratios (Iterable): octave aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        anchoring_means (Iterable): Mean values of anchoring targets.
        anchoring_stds (Iterable): Std values of anchoring targets.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        deformable_groups: (int): Group number of DCN in
            FeatureAdaption module.
        loc_filter_thr (float): Threshold to filter out unconcerned regions.
        loss_loc (dict): Config of location loss.
        loss_shape (dict): Config of anchor shape loss.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of bbox regression loss.
    """

    def __init__(
            self,
            num_classes,
            in_channels,
            feat_channels=256,
            octave_base_scale=8,
            scales_per_octave=3,
            octave_ratios=[0.5, 1.0, 2.0],
            anchor_strides=[4, 8, 16, 32, 64],
            anchor_base_sizes=None,
            anchoring_means=(.0, .0, .0, .0),
            anchoring_stds=(1.0, 1.0, 1.0, 1.0),
            target_means=(.0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0),
            deformable_groups=4,
            loc_filter_thr=0.01,
            loss_loc=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.octave_scales = octave_base_scale * np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        self.approxs_per_octave = len(self.octave_scales) * len(octave_ratios)
        self.octave_ratios = octave_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.anchoring_means = anchoring_means
        self.anchoring_stds = anchoring_stds
        self.target_means = target_means
        self.target_stds = target_stds
        self.deformable_groups = deformable_groups
        self.loc_filter_thr = loc_filter_thr
        self.approx_generators = []
        self.square_generators = []
        for anchor_base in self.anchor_base_sizes:
            # Generators for approxs
            self.approx_generators.append(
                AnchorGenerator(anchor_base, self.octave_scales,
                                self.octave_ratios))
            # Generators for squares
            self.square_generators.append(
                AnchorGenerator(anchor_base, [self.octave_base_scale], [1.0]))
        # one anchor per location
        self.num_anchors = 1
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_focal_loss = loss_cls['type'] in ['FocalLoss']
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        # build losses
        self.loss_loc = build_loss(loss_loc)
        self.loss_shape = build_loss(loss_shape)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1)
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2,
                                    1)
        self.feature_adaption = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.conv_cls = MaskedConv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.conv_reg = MaskedConv2d(self.feat_channels, self.num_anchors * 4,
                                     1)

    def init_weights(self):
        normal_init(self.conv_cls, std=0.01)
        normal_init(self.conv_reg, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_loc, std=0.01, bias=bias_cls)
        normal_init(self.conv_shape, std=0.01)

        self.feature_adaption.init_weights()

    def forward_single(self, x):
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        # masked conv is only used during inference for speed-up
        if not self.training:#在测试的时候使用，测试一次只有一张图，所以用[0]
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        #self.conv_cls和self.conv_reg是MaskedConv2d,在train的时候mask为None,self.conv_cls和self.conv_reg就是正常的nn.conv2d
        #在test的时候，mask不是None,conv_cls和conv_reg使用的是MaskedConv2d
        #在test使用mask的时候，就不会对所有的guided anchors进行conv_cls和conv_reg,所以起到加速的作用
        #loc_pred在test的时候分别在这几个时候起到作用：
        #1.conv_cls和conv_reg的时候使用MaskedConv2d输入了loc_pred的mask(注意此时输出的cls_score和bbox_pred的shape和没用mask是一样的)
        #2.在get_bboxes中的get_anchors时用mask对squares和shape_pred进行了过滤，所以输出的guided anchors和loc_mask都是过滤后的
        #3.在get_bboxes_single中对输入的cls_score和bbox_pred用mask做了过滤
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_sampled_approxs(self, featmap_sizes, img_metas, cfg):
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # approxes for one time
        multi_level_approxs = []
        for i in range(num_levels):
            approxs = self.approx_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_approxs.append(approxs)
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]

        # for each image, we compute inside flags of multi level approxes
        inside_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]
            for i in range(num_levels):
                approxs = multi_level_approxs[i]
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.approx_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                inside_flags_list = []
                for i in range(self.approxs_per_octave):
                    split_valid_flags = flags[i::self.approxs_per_octave]
                    split_approxs = approxs[i::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(
                        split_approxs, split_valid_flags,
                        img_meta['img_shape'][:2], cfg.allowed_border)
                    inside_flags_list.append(inside_flags)
                # inside_flag for a position is true if any anchor in this
                # position is true
                inside_flags = (
                    torch.stack(inside_flags_list, 0).sum(dim=0) > 0)
                multi_level_flags.append(inside_flags)
            inside_flag_list.append(multi_level_flags)
        return approxs_list, inside_flag_list

    def get_anchors(self,
                    featmap_sizes,
                    shape_preds,
                    loc_preds,
                    img_metas,
                    use_loc_filter=False):
        """Get squares according to feature map sizes and guided
        anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
                loc masks of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # squares for one time
        multi_level_squares = []
        for i in range(num_levels):
            squares = self.square_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_squares.append(squares)
        squares_list = [multi_level_squares for _ in range(num_imgs)]#rpn中的anchor都是按图像的list

        # for each image, we compute multi level guided anchors
        guided_anchors_list = []#anchors是按图像的list
        loc_mask_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]#shape_preds是一个list
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self.get_guided_anchors_single(
                    squares,
                    shape_pred,
                    loc_pred,
                    use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)#名字中有level就是按level的List
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return squares_list, guided_anchors_list, loc_mask_list

    def get_guided_anchors_single(self,
                                  squares,
                                  shape_pred,
                                  loc_pred,
                                  use_loc_filter=False):
        """Get guided anchors and loc masks for a single level.

        Args:
            square (tensor): Squares of a single level.
            shape_pred (tensor): Shape predections of a single level.
            loc_pred (tensor): Loc predections of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.

        Returns:
            tuple: guided anchors, location masks
        """
        # calculate location filtering mask
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
        mask = mask.contiguous().view(-1)
        # calculate guided anchors
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(
            -1, 2).detach()[mask]#.permute后如果接view,就需要加上contiguous,也可以直接使用reshape
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas#相当于dx,dy=0,anchor_deltas是dw,dh
        guided_anchors = delta2bbox(
            squares,
            bbox_deltas,
            self.anchoring_means,
            self.anchoring_stds,
            wh_ratio_clip=1e-6)
        return guided_anchors, mask

    #先变形把形状变一致之后,根据bbox_anchors建立bbox_deltas的new_full
    #再对bbox_deltas进行切片赋值，通过delta2bbox获得pred_anchors_，
    #再把pred和target带入iou loss中
    def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts,
                          anchor_weights, anchor_total_num):
        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        # filter out negative samples to speed-up weighted_bounded_iou_loss
        inds = torch.nonzero(anchor_weights[:, 0] > 0).squeeze(1)
        bbox_deltas_ = bbox_deltas[inds]
        bbox_anchors_ = bbox_anchors[inds]
        bbox_gts_ = bbox_gts[inds]
        anchor_weights_ = anchor_weights[inds]
        pred_anchors_ = delta2bbox(
            bbox_anchors_,
            bbox_deltas_,
            self.anchoring_means,
            self.anchoring_stds,
            wh_ratio_clip=1e-6)
        loss_shape = self.loss_shape(
            pred_anchors_,#(x,y,x,y) (k,4)
            bbox_gts_,#(x,y,x,y)     (k,4)
            anchor_weights_,
            avg_factor=anchor_total_num)
        return loss_shape

    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor,
                        cfg):
        loss_loc = self.loss_loc(
            loc_pred.reshape(-1, 1),
            loc_target.reshape(-1, 1).long(),
            loc_weight.reshape(-1, 1),
            avg_factor=loc_avg_factor)
        return loss_loc

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             shape_preds,
             loc_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.approx_generators)

        # get loc targets
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,#按图像的list
            featmap_sizes,#按level的list (h,w)
            self.octave_base_scale,#8
            self.anchor_strides,#[4, 8, 16, 32, 64]
            center_ratio=cfg.center_ratio,#0.2
            ignore_ratio=cfg.ignore_ratio)#0.5

        # get sampled approxes
        #get_sampled_approxs就是生成了按图像的List,每个里面是5个层级的anchors 每个的shape形如(k*9,4)
        approxs_list, inside_flag_list = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg)
        # get squares and guided anchors
        #get_anchors生成的squares_list是按图像的List,每个里面是5个层级的anchors 每个层的shape形如(k,4)
        #生成的guided_anchors_list是按图像的list,其中在每个图的每个层级上调用了get_guided_anchors_single生成guided_anchors和loc_mask
        #首先根据是否要use_loc_filter，用loc_pred>=loc_filter_thr或0.0来生成loc_mask，permute后带入squares中
        #定义bbox_deltas的大小为squares的大小初始化为0，再把anchor_deltas带入[:,2:]，即dw,dh的位置
        #anchor_deltas是shape_preds.permute(1,2,0).contiguous().view(-1,2).detach()[mask]
        #生成的loc_mask_list也是按图像的list
        squares_list, guided_anchors_list, _ = self.get_anchors(
            featmap_sizes, shape_preds, loc_preds, img_metas)

        # get shape targets
        sampling = False if not hasattr(cfg, 'ga_sampler') else True
        #ga_shape_target在guided anchor_target中
        #输入approxs_list和inside_flag_list,squares_list,gt_bboxes等
        #首先把一个图像的approxs_list和inside_flag_list和squares_list cat到一个tensro
        #调用ga_shape_target_single,构建assigner(approx_max_iou_assigner),在assign中
        #先是把approxs view成(num_squares,9,4)即(k,9,4)再变成(9,k,4)最后变成(9*k,4)
        #再对approxs和gt使用bbox_overlaps求出每个anchor的iou，再view(9,k,num_gts).max(dim=0)
        #求出每个图像5层的approxs的每个位置上9个anchor的最大iou的值，再transpose成(num_gts,num_squares)
        #这里在每个位置使用9个anchor来做assign的原因是：assign的结果会用来做loss，如果单纯只用squares
        #因为squares的形状单一，可能有一些比较好的位置的squares会被assign成负样本，这样训练的时候也会被当成是负样本训练
        #这样在生成guided_anchors的时候该位置就不会有anchors了，但实际上该位置的squares当形状改变时会有比较好的结果，所以
        #就用9个anchors来做assign，为了保证这些有潜力的squares会被留下当成是正样本训练，在生成guided_anchors的时候也会生成比较好的guided_anchors
        #再使用assign_wrt_overlaps去求出AssignResult
        #再sample,注意此时sample的输入是根据approxs生成的AssignResult和squares
        #定义bbox_anchors,bbox_gts,bbox_weights,把sampling_result.pos_inds作为索引值带入
        #这三个中，赋值为sampling_result.pos_bboxes和sampling_result.pos_gt_bboxes和1.0
        #再求出所有图像的被采样后的正样本数量和以及负样本数量和，最后再按level生成List,这是为了和pre的形状符合，便于loss
        shape_targets = ga_shape_target(
            approxs_list,
            inside_flag_list,
            squares_list,
            gt_bboxes,
            img_metas,
            self.approxs_per_octave,
            cfg,
            sampling=sampling)
        if shape_targets is None:
            return None
        #按层级的list,其中bbox_anchors_list每一个是(2,k,4)，这些由anchor生成的都不具备h,w形状所以是k,而pre有h,w的形状
        #bbox_gts_list和anchor_weights_list每一个也都是(2,k,4)
        #bbox_anchors_list的pos的位置是squares的(x,y,x,y)的坐标；bbox_gts_list是对应的gt的坐标，这才是target,而bbox_anchors_list是结合shap_pre生成坐标预测用的
        (bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num,
         anchor_bg_num) = shape_targets 
        anchor_total_num = (
            anchor_fg_num if not sampling else anchor_fg_num + anchor_bg_num)

        # get anchor targets
        #anchor_targe和shape_targets类似，和在普通rpn中生成anchor_target一样，最后输出的
        #是一个tuple,里面有labels_list, label_weights_list, bbox_targets_list,bbox_weights_list, num_total_pos, num_total_neg
        #这些也都是做loss_cls和loss_bbox要用的
        #按层级的list,其中labels_list每一个是(2,k)
        #bbox_targets_list和bbox_weights_list每一个也都是(2,k,4) 
        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            guided_anchors_list,
            inside_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos if self.cls_focal_loss else num_total_pos +
            num_total_neg)

        # get classification and bbox regression losses
        #以层级为单位，每个level中cls_scores为(2,1,h,w);bbox_preds为(2,4,h,w);labels_list为(2,k);bbox_targets_list为(2,k,4)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,#在继承的anchor_head中
            cls_scores,#(2,1,h,w)
            bbox_preds,#(2,4,h,w)
            labels_list,#(2,k)
            label_weights_list,#(2,k)
            bbox_targets_list,#(2,k,4)
            bbox_weights_list,#(2,k,4)
            num_total_samples=num_total_samples,#num_total_samples是所有图像的所有level的正负样本数量和，是因为求loss是按照level来的，每次level会求一个batch的Loss,之后再把所有的level的loss相加，这样就相当于是全都相加再除以总的num_total_samples
            cfg=cfg)

        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],#(2,1,h,w)
                loc_targets[i],#(2,1,h,w)
                loc_weights[i],#(2,1,h,w)
                loc_avg_factor=loc_avg_factor,
                cfg=cfg)
            losses_loc.append(loss_loc)

        # get anchor shape loss
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(
                shape_preds[i],#(2,2,h,w) 是相对值dw,dh
                bbox_anchors_list[i],#(2,k,4) 是绝对的坐标值 所以在该函数里面需要把
                bbox_gts_list[i],#(2,k,4)
                anchor_weights_list[i],#(2,k,4)
                anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_shape=losses_shape,
            loss_loc=losses_loc)

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'shape_preds', 'loc_preds'))
    def get_bboxes(self,
                   cls_scores,#rpn输出的pred都是按level的list
                   bbox_preds,#rpn输出的pred都是按level的list
                   shape_preds,#rpn输出的pred都是按level的list
                   loc_preds,#rpn输出的pred都是按level的list
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(
            loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # get guided anchors
        _, guided_anchors, loc_masks = self.get_anchors(
            featmap_sizes,
            shape_preds,
            loc_preds,
            img_metas,
            use_loc_filter=not self.training)
        result_list = []
        for img_id in range(len(img_metas)):#对图像操作，取出cls_score_list，bbox_pred_list，guided_anchor_list，loc_mask_list的同一副图像下5个level的值
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            guided_anchor_list = [
                guided_anchors[img_id][i].detach() for i in range(num_levels)
            ]
            loc_mask_list = [
                loc_masks[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               guided_anchor_list,
                                               loc_mask_list, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list
    #下面的函数在ga_rpn_head中有定义
    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          mlvl_masks,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        #一次迭代的cls_scores:(1,h,w);bbox_preds:(4,h,w);mlvl_anchors：(k,4) 其中k=h*w ;mlvl_masks:(k,1)
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       mlvl_anchors,
                                                       mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and bbox_pred
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
