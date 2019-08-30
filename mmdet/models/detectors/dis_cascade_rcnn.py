from __future__ import division

import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, build_assigner, build_sampler,
                        merge_aug_masks)
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from ..builder import build_loss

@DETECTORS.register_module
class DisCascadeRCNN(BaseDetector, RPNTestMixin):#DistributedCascadeRCNN

    def __init__(self,
                 num_stages,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        super(DisCascadeRCNN, self).__init__()

        self.num_stages = num_stages
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = nn.ModuleList()
            self.bbox_head = nn.ModuleList()
            if not isinstance(bbox_roi_extractor, list):
                bbox_roi_extractor = [
                    bbox_roi_extractor for _ in range(num_stages)
                ]
            if not isinstance(bbox_head, list):
                bbox_head = [bbox_head for _ in range(num_stages)]
            assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
            for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
                self.bbox_roi_extractor.append(
                    builder.build_roi_extractor(roi_extractor))
                self.bbox_head.append(builder.build_head(head))
            
            self.iou_head = nn.ModuleList()
            for i in range(2):
                fc_in_channels = (
                    256 * 7 * 7 if i == 0 else 1024) #第一个fc的输入是self.in_channels * self.roi_feat_area（这个在上面有定义）
                self.iou_head.append(nn.Linear(fc_in_channels, 1024))
            self.relu = nn.ReLU(inplace=True)
            self.iou_cls = nn.Linear(1024, 3)
            loss_iou = dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.1
            )
            self.loss_iou = build_loss(loss_iou)


        if mask_head is not None:
            self.mask_head = nn.ModuleList()
            if not isinstance(mask_head, list):
                mask_head = [mask_head for _ in range(num_stages)]
            assert len(mask_head) == self.num_stages
            for head in mask_head:
                self.mask_head.append(builder.build_head(head))
            if mask_roi_extractor is not None:
                self.share_roi_extractor = False
                self.mask_roi_extractor = nn.ModuleList()
                if not isinstance(mask_roi_extractor, list):
                    mask_roi_extractor = [
                        mask_roi_extractor for _ in range(num_stages)
                    ]
                assert len(mask_roi_extractor) == self.num_stages
                for roi_extractor in mask_roi_extractor:
                    self.mask_roi_extractor.append(
                        builder.build_roi_extractor(roi_extractor))
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    def init_weights(self, pretrained=None):
        super(DisCascadeRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
            if self.with_mask:
                if not self.share_roi_extractor:
                    self.mask_roi_extractor[i].init_weights()
                self.mask_head[i].init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox heads
        rois = bbox2roi([proposals])
        if self.with_bbox:
            for i in range(self.num_stages):
                bbox_feats = self.bbox_roi_extractor[i](
                    x[:self.bbox_roi_extractor[i].num_inputs], rois)
                if self.with_shared_head:
                    bbox_feats = self.shared_head(bbox_feats)
                cls_score, bbox_pred = self.bbox_head[i](bbox_feats)
                outs = outs + (cls_score, bbox_pred)
        # mask heads
        if self.with_mask:
            mask_rois = rois[:100]
            for i in range(self.num_stages):
                mask_feats = self.mask_roi_extractor[i](
                    x[:self.mask_roi_extractor[i].num_inputs], mask_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                mask_pred = self.mask_head[i](mask_feats)
                outs = outs + (mask_pred, )
        return outs

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img)

        losses = dict()

        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        
        bbox_assigner = build_assigner(self.train_cfg.rcnn[0].assigner)
        bbox_roi_extractor = self.bbox_roi_extractor[0]
        proposal_one_stage = []
        proposal_two_stage = []
        proposal_three_stage = []
        proposal_stages = []
        num_imgs = img.size(0)
        for j in range(num_imgs):
            one_stage,two_stage,three_stage = bbox_assigner.get_proposal_iou(
                                                proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                                                gt_labels[j])
            proposal_one_stage.append(one_stage)
            proposal_two_stage.append(two_stage)
            proposal_three_stage.append(three_stage)
        proposal_stages.append(proposal_one_stage)
        proposal_stages.append(proposal_two_stage) 
        proposal_stages.append(proposal_three_stage) 

        rois1 = bbox2roi(proposal_one_stage)
        rois2 = bbox2roi(proposal_two_stage)
        rois3 = bbox2roi(proposal_three_stage)
        rois = torch.cat([rois1, rois2, rois3],dim = 0)
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois) 
        gt_iou1 = rois.new_zeros((rois1.size(0),),dtype=torch.long)
        gt_iou2 = rois.new_full((rois2.size(0),),1,dtype=torch.long)
        gt_iou3 = rois.new_full((rois3.size(0),),2,dtype=torch.long)
        gt_iou = torch.cat([gt_iou1, gt_iou2, gt_iou3])
        iou_x = bbox_feats.view(bbox_feats.size(0),-1)
        for fc in self.iou_head:
            iou_x = self.relu(fc(iou_x))
        iou_cls = self.iou_cls(iou_x)

        losses['loss_iou'] = self.loss_iou(
            iou_cls,
            gt_iou,
            avg_factor=iou_cls.size(0)
        )
        
        #3次的迭代在assign和proposals之间
        for i in range(self.num_stages):
            proposal_stage = proposal_stages[i]
            if i == 0:
                proposal_list = proposal_stage
            else:
                pro_list = []
                for j in range(num_imgs):
                    pro= torch.cat([proposal_list[j],proposal_stage[j]],dim = 0)
                    pro_list.append(pro)
                proposal_list = pro_list
            #train_cfg.rcnn，lw,roi_extractor,bbox_head都有三个
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg.rcnn[i]
            lw = self.train_cfg.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            if self.with_bbox or self.with_mask: #进入
                bbox_assigner = build_assigner(rcnn_train_cfg.assigner)
                bbox_sampler = build_sampler(
                    rcnn_train_cfg.sampler, context=self)
                num_imgs = img.size(0)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]

                for j in range(num_imgs):
                    assign_result = bbox_assigner.assign(
                        proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
                        gt_labels[j])
                    sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[j],
                        gt_bboxes[j],
                        gt_labels[j],
                        feats=[lvl_feat[j][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)

            # bbox head forward and loss
            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                            rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats)

            bbox_targets = bbox_head.get_target(sampling_results, gt_bboxes,
                                                gt_labels, rcnn_train_cfg)
            loss_bbox = bbox_head.loss(cls_score, bbox_pred, *bbox_targets)
            for name, value in loss_bbox.items():
                losses['s{}.{}'.format(i, name)] = (
                    value * lw if 'loss' in name else value)

            # mask head forward and loss
            if self.with_mask:
                if not self.share_roi_extractor:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    pos_rois = bbox2roi(
                        [res.pos_bboxes for res in sampling_results])
                    mask_feats = mask_roi_extractor(
                        x[:mask_roi_extractor.num_inputs], pos_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                else:
                    # reuse positive bbox feats
                    pos_inds = []
                    device = bbox_feats.device
                    for res in sampling_results:
                        pos_inds.append(
                            torch.ones(
                                res.pos_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                        pos_inds.append(
                            torch.zeros(
                                res.neg_bboxes.shape[0],
                                device=device,
                                dtype=torch.uint8))
                    pos_inds = torch.cat(pos_inds)
                    mask_feats = bbox_feats[pos_inds]
                mask_head = self.mask_head[i]
                mask_pred = mask_head(mask_feats)
                mask_targets = mask_head.get_target(sampling_results, gt_masks,
                                                    rcnn_train_cfg)
                pos_labels = torch.cat(
                    [res.pos_gt_labels for res in sampling_results])
                loss_mask = mask_head.loss(mask_pred, mask_targets, pos_labels)
                for name, value in loss_mask.items():
                    losses['s{}.{}'.format(i, name)] = (
                        value * lw if 'loss' in name else value)

            # refine bboxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results] #按图像的list
                roi_labels = bbox_targets[0]  # bbox_targets是tuple 序号零对应的是label
                with torch.no_grad():
                    proposal_list = bbox_head.refine_bboxes(
                        rois, roi_labels, bbox_pred, pos_is_gts, img_meta) #proposal_list依旧是按照图像来的

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        x = self.extract_feat(img)
        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        img_shape = img_meta[0]['img_shape']
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        # "ms" in variable names means multi-stage
        ms_bbox_result = {}
        ms_segm_result = {}
        ms_scores = []
        rcnn_test_cfg = self.test_cfg.rcnn
        #在test的时候bbox2roi不需要被循环，是因为test的时候不用assigin和sample,所以也就不用生成
        #按照图像分的proposals，就直接按照rois来就行，调用的也是regress_by_class而不是refine_bboxes
        #refine_bboxes(这个是在用regress_by_class生成rois的基础上用把图像和图像分开生成proposals的)
        rois = bbox2roi(proposal_list)

        bbox_roi_extractor = self.bbox_roi_extractor[0]
        bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
        iou_x = bbox_feats.view(bbox_feats.size(0),-1)
        for fc in self.iou_head:
            iou_x = self.relu(fc(iou_x))
        iou_cls = self.iou_cls(iou_x)   
        inds = torch.argmax(iou_cls,dim = 1) 
        rois1 = rois[inds==0]
        rois2 = rois[inds==1]
        rois3 = rois[inds==2]
        rois_all = [rois1,rois2,rois3]
        for i in range(self.num_stages):

            rois_stage = rois_all[i]
            if i == 0:
                rois = rois_stage
            else:
                rois = torch.cat([rois,rois_stage],dim=0)

            bbox_roi_extractor = self.bbox_roi_extractor[i]
            bbox_head = self.bbox_head[i]

            bbox_feats = bbox_roi_extractor(
                x[:len(bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = bbox_head(bbox_feats) #(n,81) (n,4)
            ms_scores.append(cls_score)

            if self.test_cfg.keep_all_stages:
                det_bboxes, det_labels = bbox_head.get_det_bboxes(
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                bbox_result = bbox2result(det_bboxes, det_labels,
                                          bbox_head.num_classes)
                ms_bbox_result['stage{}'.format(i)] = bbox_result

                if self.with_mask:
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_head = self.mask_head[i]
                    if det_bboxes.shape[0] == 0:
                        mask_classes = mask_head.num_classes - 1
                        segm_result = [[] for _ in range(mask_classes)]
                    else:
                        _bboxes = (
                            det_bboxes[:, :4] *
                            scale_factor if rescale else det_bboxes)
                        mask_rois = bbox2roi([_bboxes])
                        mask_feats = mask_roi_extractor(
                            x[:len(mask_roi_extractor.featmap_strides)],
                            mask_rois)
                        if self.with_shared_head:
                            mask_feats = self.shared_head(mask_feats, i)
                        mask_pred = mask_head(mask_feats)
                        segm_result = mask_head.get_seg_masks(
                            mask_pred, _bboxes, det_labels, rcnn_test_cfg,
                            ori_shape, scale_factor, rescale)
                    ms_segm_result['stage{}'.format(i)] = segm_result

            if i < self.num_stages - 1:
                bbox_label = cls_score.argmax(dim=1)
                rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred,
                                                  img_meta[0]) #(n,5)

        # cls_score = sum(ms_scores) / self.num_stages #在test的时候rois的个数不变，分数是各阶段的sum/阶段数,分数就在最后get_det_bboxes的时候才用到
        cls_score1 = (ms_scores[0][:len(ms_scores[0])] + ms_scores[1][:len(ms_scores[0])] + ms_scores[2][:len(ms_scores[0])])/self.num_stages
        cls_score2 = (ms_scores[1][len(ms_scores[0]):] + ms_scores[2][len(ms_scores[0]):(len(ms_scores[1]))])/(self.num_stages-1)
        cls_score3 = ms_scores[2][len(ms_scores[1]):]/(self.num_stages-2)
        cls_score = torch.cat([cls_score1, cls_score2, cls_score3])
        det_bboxes, det_labels = self.bbox_head[-1].get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)  #det_bboxes是(k,5) det_labels是(k,)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.bbox_head[-1].num_classes) #把bbox和label .cpu().numpy()
        #bbox_result是list，按照类别分，这就是输入det_labels的目的，det_labels是(0,79),此时就不管背景了
        ms_bbox_result['ensemble'] = bbox_result

        if self.with_mask:
            if det_bboxes.shape[0] == 0:
                mask_classes = self.mask_head[-1].num_classes - 1
                segm_result = [[] for _ in range(mask_classes)]
            else:
                if isinstance(scale_factor, float):  # aspect ratio fixed
                    _bboxes = (
                        det_bboxes[:, :4] *
                        scale_factor if rescale else det_bboxes)
                else:
                    _bboxes = (
                        det_bboxes[:, :4] *
                        torch.from_numpy(scale_factor).to(det_bboxes.device)
                        if rescale else det_bboxes)

                mask_rois = bbox2roi([_bboxes])
                aug_masks = []
                for i in range(self.num_stages):
                    mask_roi_extractor = self.mask_roi_extractor[i]
                    mask_feats = mask_roi_extractor(
                        x[:len(mask_roi_extractor.featmap_strides)], mask_rois)
                    if self.with_shared_head:
                        mask_feats = self.shared_head(mask_feats)
                    mask_pred = self.mask_head[i](mask_feats)
                    aug_masks.append(mask_pred.sigmoid().cpu().numpy())
                merged_masks = merge_aug_masks(aug_masks,
                                               [img_meta] * self.num_stages,
                                               self.test_cfg.rcnn)
                segm_result = self.mask_head[-1].get_seg_masks(
                    merged_masks, _bboxes, det_labels, rcnn_test_cfg,
                    ori_shape, scale_factor, rescale)
            ms_segm_result['ensemble'] = segm_result

        if not self.test_cfg.keep_all_stages:
            if self.with_mask:
                results = (ms_bbox_result['ensemble'],
                           ms_segm_result['ensemble'])
            else:
                results = ms_bbox_result['ensemble']
        else:
            if self.with_mask:
                results = {
                    stage: (ms_bbox_result[stage], ms_segm_result[stage])
                    for stage in ms_bbox_result
                }
            else:
                results = ms_bbox_result

        return results

    def aug_test(self, img, img_meta, proposals=None, rescale=False):
        raise NotImplementedError

    def show_result(self, data, result, img_norm_cfg, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        super(DisCascadeRCNN, self).show_result(data, result, img_norm_cfg,
                                             **kwargs)
