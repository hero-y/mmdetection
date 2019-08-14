import torch

from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from mmdet.core import bbox2roi, build_assigner, build_sampler

"""
一个类，继承TwoStageDetector,里面有三个函数，初始化,forward_train,simple_test_bboxes
在rpn中先把5个特征图输入到rpn网络中，再把其输出+img_meta+self.train_cfg.rpn+gt_bboxes作为输入放到Loss中
再把其输出+proposal的cfg+img_meta作为输入输入到get_bboxes中，这两个相加都是赋给了一个值，代入函数的时候用*
生成的proposals是按图像的list
接下来对Proposals进行assign_and_sample,分别构造assign和sample的实例，构建sampling_results的lsit，按图像进行for in
把proposal和gt_bboxes和gt_labels放入assign中，再把assign_result和proposals和gt_bboxes和gt_labels放入sample
最后bbox2roi,再提取roi的特征图，再target最后loss
"""
@DETECTORS.register_module
class DoubleHeadRCNN(TwoStageDetector):

    def __init__(self, reg_roi_scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.reg_roi_scale_factor = reg_roi_scale_factor

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        x = self.extract_feat(img) #img用的是img_meta中的pad_shape的大小 这个时候img已经在cuda上了,不是list,是(n,3,h,w)，len(img)=n
        
        losses = dict()
        #gt_bboxes:len(gt_bboxes)=图像数量
        #gt_labels,len(gt_labels)=图像数量  [tensor([15, 16, 15], device='cuda:0'), tensor([61, 54, 46], device='cuda:0')]
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)  #是个List，按层来的
            #img_meta [{'ori_shape': (429, 640, 3), 'img_shape': (201, 300, 3), 'pad_shape': (224, 320, 3), 'scale_factor': 0.46875, 'flip': False}, {'ori_shape': (427, 640, 3), 'img_shape': (200, 300, 3), 'pad_shape': (224, 320, 3), 'scale_factor': 0.46875, 'flip': True}]
            #'ori_shape'指的是图像原始的尺寸大小，img_shape是缩放到cfg中设定的大小后的图像大小，pad_shape是对img_shape进行微调，使得图像的大小的h,w符合2的倍数(16,32等)，scale_factor是img_shape/ori_shape
            #图像最开始是(h,w,c),经过卷积之后(c,h,w)
            #gt_bboxes_ignore [tensor([], device='cuda:0', size=(0, 4)), tensor([], device='cuda:0', size=(0, 4))]
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)  #rpn_losses也是个dict

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)  #得到的是List,list的对象是img,每个对象的内容是总proposal,大小为(n,5),前四个是坐标，最后一个是scores，要有score，因为下面对proposal进行sample的时候要用score
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None: #没有进来
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []  #都是把图像放到一个List里面，[i]实际是把第i个图像的相关信息取出来
            for i in range(num_imgs):
                #proposal_list (n,5)[:4]是坐标，第5个值是score
                #device都是cuda,也都是tensor
                #gt_bboxes_ignore:  tensor([], device='cuda:0', size=(0, 4))
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            #sample.bbox的里面是正样本+负样本，这个时候的bbox就是坐标信息了，没有score信息了，因为不需要了
            #bbox2roi也就是要变成(n,5),不但要把两个图像的bbox拼到一起，还要再加一个img_ind的信息
            #先用torch.cat(,dim=-1),出来[batch_ind, x1, y1, x2, y2]，两个图像放到list中，再用一次torch.cat(dim=0)
            rois = bbox2roi([res.bboxes for res in sampling_results])  #输入的是list，list的序号是图像的序号
            #res.bboxes正负样本都有bbox的坐标，只不过负样本没有gt_bbox
            # TODO: a more flexible way to decide which feature maps to use
            #知道原图上的坐标后，找到对应的特征图，再把对应的位置给取出来
            bbox_cls_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)  #self.bbox_roi_extractor.num_inputs=5
            bbox_reg_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs],
                rois,
                roi_scale_factor=self.reg_roi_scale_factor)
            if self.with_shared_head: #没有跳进来
                bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_cls_feats,
                                                  bbox_reg_feats)
            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)  #loss_bbox是一个字典，字典有.update()的功能，类似list的append

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_cls_feats.device
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
                mask_feats = bbox_cls_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        return losses

    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
