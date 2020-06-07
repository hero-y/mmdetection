import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob

INF = 1e8 #e代表指数，1e8 = 1*10的8次幂

"""
FCOS和RetinaNet中使用的FocalLoss,类别输出的通道数都是80，而FasterRCNN的CrossEntropyLoss的类别输出通道数是81
FocalLoss使用的是sigmoid版本的交叉熵函数，根据label，把80个通道的每一个都设置为0或1，即二分类，分类出每一个通道是不是label对应的。
最后的Loss是对每一个通道都求了Log。
"""
@HEADS.register_module
class FCOSHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(FCOSHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()
        return cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes,
                                                gt_labels)
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        #既然target变成了按照level的list，且如bbox_targets来说，其中每个level的shape是(k,4),k包含了一个batch中所有图像在该level的点数
        #故pred也要改变shape, .permute(0,2,3,1).reshape(-1,4)之所以把第一维通道放到最后是因为第一维通道就是bbox的4个值
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        #对pred和target都把5个level的值cat到一起
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        #根据安排的label的情况确定pos和neg
        #该pos_inds是针对一个batch的所有level而言的
        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        #focal_loss没有sample的过程，avg_factor是num_pos，是因为focal_loss本身就是处理正负样本不均衡，所以不用sample成3:1,
        #而对于avg_factor因为比较容易的负样本在focal_loss中都已经被抑制了，所以就用num_pos即可
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            #根据cell的坐标点和pre以及target的l,r,t,d求出pre和target的左上角和右下角的坐标
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            #iou_loss是通过4个坐标点求出pre和target的iou，再-log()即可
            #但对于l,r,t,d四个量,我认为回归的效果应该没有使用anchor的效果好，因为anchor中回归的是dx,dy,dw,dh
            #可能是因为iou_loss比smooth_l1_loss效果要好，权重使用的是pos_centerness_targets,意思就是,对于某些
            #对偏离gt中心比较远的cell的iou_loss设置的权重小一点，avg_factor是pos_centerness_targets.sum()
            #这是把所有的pos_centerness_targets的值相加，作为avg_factor，因为每个cell的权重设置的不一样，就不能只是用pos的数量作为avg_factor
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum()) 
            #计算centerness的loss使用BinaryCrossEntropyWithLogits,该Loss是对于二分类用的交叉熵函数
            #内部集成了sigmoid，而BinaryCrossEntropy是没有继承sigmoid的。BCELoss的输入的target可以是
            #任意值，不一定是0,1,但输出的通道数只能是1个。且输入的target需要是FloatTensor，对于CrossEntropyLoss
            #内部集成了softmax,是对多分类使用的，故bbox_head中输出的类别通道一直都是C个，anchor_head中用的也是
            #BCELoss,是在cfg中，type是CrossEntropyLoss,因为有sigmoid=True，所以也就是BCELoss
            #在这里没有使用avg_factor，可能是因为loss值本身较小？
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            #scores和centerness再permute和reshape之后都要sigmoid
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        #在这里使用padding，给scores添加一维的目的是,因为multiclass_nms中是根据类别进行nms
        #是range(1,81),即只对前景类进行nms,背景不管,因为在faster rcnn中类别输出是81，而在单阶段中
        #输出是80,所以为了不改动nms的代码,故给scores添加第一维
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,#对于dict来说正常是不能用.取键值的,但这里可能是因为最开始用Config.fromfile，后面所有的cfg都可以用.的方式取了
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    """
    获取每个level特征图的每个点在原图上的坐标，返回值为[h*w,2]
    通过先设定一维的x_range和y_range再通过torch.meshgrid输出两个二维的值(跳进去看源码)
    再通过torch.stack把x.reshape(-1)和y.reshape(-1)对应起来
    """
    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        #stride//2：因为使用上述方法，其实是获得了每个point对应在原图感受野的左上角的坐标
        #所以+stride//2就是获得该感受野的中心点的坐标值
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2 
        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]#5个level的list,每个值的shape是(num_points,2),5个level是有5种regress_ranges
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        #multi_apply的输入参数也不一定都是List,不是List的就重复使用了
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        #实现按level的list
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1) #(num_points,num_gts)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2) #(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0] #(num_points,num_gts)
        right = gt_bboxes[..., 2] - xs #(num_points,num_gts)
        top = ys - gt_bboxes[..., 1] #(num_points,num_gts)
        bottom = gt_bboxes[..., 3] - ys #(num_points,num_gts)
        bbox_targets = torch.stack((left, top, right, bottom), -1) #(num_points,num_gts,4)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0 #(num_points,num_gts),其中-1代表最后一维，即4，比较4个坐标中的最小值，min(-1)返回的是values,indices，[0]代表取出values

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1]) #(num_points,num_gts)

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        #根据求出的inside_regress_range和inside_gt_bbox_mask把其中是0的gt的面积赋值为一个很大的数INF
        #再求出每个位置的最小面积的gt，对该gt做回归
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        #min_area和min_area_inds的shape都是[num_points]
        min_area, min_area_inds = areas.min(dim=1)

        #min_area_inds是一维的，可以直接代入gt_labels中
        #target和target之间应该是有关联的，如这里是对每一个点穷举出所有的gt,求出和每个gt的target，
        #再根据大小做出过滤，过滤后生成的序号可以用来求出对应的label_target,这里用的是area,个人人为
        #应该使用centerness的思想，将最大的centerness对应的gt的label作为target
        labels = gt_labels[min_area_inds] #(num_points,1)
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds] #(num_points,4)

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
