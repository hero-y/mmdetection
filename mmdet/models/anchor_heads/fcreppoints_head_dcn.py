import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob
from mmdet.core import PointGenerator
from mmdet.ops import ModulatedDeformConvPack
from mmdet.models.utils import build_norm_layer
INF = 1e8 #e代表指数，1e8 = 1*10的8次幂

@HEADS.register_module
class FCRepPointsDCNHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 num_points=9,
                 gradient_mul=0.1,
                 strides=(4, 8, 16, 32, 64),
                 point_base_scale=4,
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
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 transform_method='minmax'):
        super(FCRepPointsDCNHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.transform_method = transform_method
        self.point_generators = [PointGenerator() for _ in self.strides]
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i == self.stacked_convs-1:
                self.cls_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.cls_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.cls_convs.append(nn.ReLU(inplace=True))

                self.reg_convs.append(
                    ModulatedDeformConvPack(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                if self.norm_cfg:
                    self.reg_convs.append(build_norm_layer(self.norm_cfg, self.feat_channels)[1])
                self.reg_convs.append(nn.ReLU(inplace=True))
            
            else:
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
                        bias=self.norm_cfg is None)
                )
                self.reg_convs.append(
                    ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None)
                )
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.fcos_reg = nn.Conv2d(self.feat_channels, self.num_points*2, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for i in range(len(self.cls_convs)):
            if i == self.stacked_convs-1:#只能用self.stached_convs，因为最后的Norm和relu没有和deconv合成一体
                pass
            elif i<self.stacked_convs-1:
                normal_init(self.cls_convs[i].conv, std=0.01)

        for i in range(len(self.reg_convs)):
            if i == self.stacked_convs-1:
                pass
            elif i<self.stacked_convs-1:
                normal_init(self.reg_convs[i].conv, std=0.01)
                
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def points2bbox(self, pts, y_first=True):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)
        else:
            raise NotImplementedError
        return bbox

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
        pts_out = self.fcos_reg(reg_feat)
        return cls_score, pts_out, centerness

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate.
        """
        pts_list = []
        for i_lvl in range(len(self.strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)#(n,18)注意这里要使用一下repeat因为最开始每个位置只有一个点
                pts_shift = pred_list[i_lvl][i_img]#(18,h,w)
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]#预测的点是先预测的y,再x
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.strides[i_lvl] + pts_center #注意点预测的偏移*point_strides
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            # print("pts_lvl",pts_lvl.shape)
            pts_list.append(pts_lvl)
        return pts_list#[lvl1,lvl2,lvl3,lvl4,lvl5]其中lvl1是(2,n,18)

    #  @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
            cls_scores,
            pts_outs,
            centernesses,
            gt_bboxes,
            gt_labels,
            img_metas,
            cfg,
            gt_bboxes_ignore=None):
        assert len(cls_scores) == len(pts_outs) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, pts_outs[0].dtype,
                                           pts_outs[0].device)
        labels, bbox_targets, bbox_targets_for_rep = self.fcos_target(
            all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_scores[0].size(0)
        # print("num_img",num_img)
        all_img_level_points = []
        for _ in range(num_imgs):
            all_img_level_points.append(all_level_points)
        pts_coordinate_preds = self.offset_to_pts(all_img_level_points, pts_outs) #对点×stride+中心
        bbox_preds_list = []
        for pts_coordinate_pred in pts_coordinate_preds:
            bbox_pred = self.points2bbox(
                pts_coordinate_pred.reshape(-1, 2 * self.num_points), y_first=False)
            bbox_preds_list.append(bbox_pred)
        # print("bbox_preds_list",bbox_preds_list[0].shape,bbox_preds_list[1].shape,bbox_preds_list[2].shape,bbox_preds_list[3].shape)
        flatten_cls_scores = [
            cls_score.permute(0,2,3,1).reshape(-1,self.cls_out_channels)
            for cls_score in cls_scores
        ]
        # print("flatten_cls_scores",flatten_cls_scores[0].shape,flatten_cls_scores[1].shape,flatten_cls_scores[2].shape,flatten_cls_scores[3].shape)
        flatten_centerness = [
            centerness.permute(0,2,3,1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(bbox_preds_list)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())

            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)
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
        labels_list, bbox_targets_list,bbox_targets_for_rep_list  = multi_apply(
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
        bbox_targets_for_rep_list = [
            bbox_targets_for_rep.split(num_points, 0)
            for bbox_targets_for_rep in bbox_targets_for_rep_list
        ]

        # concat per level image
        #实现按level的list
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bbox_targets_for_rep = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
            concat_lvl_bbox_targets_for_rep.append(
                torch.cat([bbox_targets_for_rep[i] for bbox_targets_for_rep in bbox_targets_for_rep_list])
            )
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_bbox_targets_for_rep

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))
        gt_bboxes_copy = gt_bboxes
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
        areas[inside_gt_bbox_mask == 0] = INF#通过给不符合的值赋值最大，来从反面
        areas[inside_regress_range == 0] = INF
        #min_area和min_area_inds的shape都是[num_points]
        min_area, min_area_inds = areas.min(dim=1)#求出的是对每个位置点来说，最合适的gt是第几个

        #min_area_inds是一维的，可以直接代入gt_labels中
        #target和target之间应该是有关联的，如这里是对每一个点穷举出所有的gt,求出和每个gt的target，
        #再根据大小做出过滤，过滤后生成的序号可以用来求出对应的label_target,这里用的是area,之所以用area
        #原因是每个位置即使是同一个level大小，也可能落入不同的物体中，但一个位置应该只回归一个物体，所以
        #选择较小的那个物体，根据物体的面积来选
        labels = gt_labels[min_area_inds] #(num_points,)
        labels[min_area == INF] = 0
        # bbox_targets_for_rep和bbox_targets不需要像labels[min_area == INF] = 0一样，因为后文求出了Label的非零序号来知道pos_inds
        bbox_targets_for_rep = gt_bboxes[range(num_points), min_area_inds]##(num_points,4)
        bbox_targets = bbox_targets[range(num_points), min_area_inds] #(num_points,4)
        # print("labels",labels.shape,"bbox_targets_for_rep",bbox_targets_for_rep.shape,"bbox_targets",bbox_targets.shape)
        return labels, bbox_targets, bbox_targets_for_rep

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        num_imgs = len(img_metas)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        img_level_points = []
        for _ in range(num_imgs):
            img_level_points.append(mlvl_points)

        pts_coordinate_preds = self.offset_to_pts(img_level_points, bbox_preds)#[lvl1,lvl2,lvl3,lvl4,lvl5]其中lvl1是(1,n,18)
        bbox_pred_tmp_list = []
        for pts_coordinate_pred in pts_coordinate_preds:
            bbox_pred = self.points2bbox(
                pts_coordinate_pred.reshape(-1, 2 * self.num_points), y_first=False)
            bbox_pred_tmp_list.append(bbox_pred)##[lvl1,lvl2,lvl3,lvl4,lvl5]其中lvl1是(n,4)
        # print("bbox_pred_tmp_list",len(bbox_pred_tmp_list),bbox_pred_tmp_list[0].shape,bbox_pred_tmp_list[1].shape)
        # print("cls_scores",cls_scores[0].shape,cls_scores[1].shape)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach()
                for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_pred_tmp_list[i].detach() for i in range(num_levels)
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
            cls_scores,bbox_preds,centernesses,mlvl_points):
            # assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1,2,0).reshape(-1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 2*self.num_points)
            # print("scores",scores.shape,"centerness",centerness.shape,"bbox_pred",bbox_pred.shape)
            nms_pre = cfg.get('nms_pre',-1)

            if nms_pre >0 and scores.shape[0]>nms_pre:
                max_scores,_ =  (scores * centerness[:,None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            mlvl_bboxes.append(bbox_pred)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
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

            