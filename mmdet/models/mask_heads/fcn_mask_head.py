import mmcv
import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule

"""
head中4个主要的函数：先初始化构建网络，再forward使用网络，再获得target，再使用loss
"""
@HEADS.register_module
class FCNMaskHead(nn.Module):

    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)):
        super(FCNMaskHead, self).__init__()
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.num_convs = num_convs
        # WARN: roi_feat_size is reserved and not used
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':#ConvTranspose2d(256,256,kernel_size=(2,2),stride=(2,2))加入转置卷积把roi从14x14变成28x28是为了增加分辨率，没其他原因
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio) #转置卷积的公式为output = (input-1)*stride+kernel_size+output_padding-2*padding (output_padding和padding默认都是0)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1) #mask rcnn中：Conv2d(256, 81, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    def init_weights(self):
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    #4个conv+1个ConvTranspose2d+1*1conv
    @auto_fp16()
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            if self.upsample_method == 'deconv':
                x = self.relu(x)
        mask_pred = self.conv_logits(x)
        return mask_pred

    #获得正proposal的target，负proposal没有mask_target，因为负的proposal没有目标bbox,而目标bbox和目标mask是关联的
    #调用mmdet/core/mask中的mask_target，输入正proposal的坐标，每个正proposal对应的label和mask和cfg
    #从输入的mask中根据每个正proposal对应的label知道应该是哪个mask,再根据输入正proposal的坐标获得该mask的bbox坐标位置的mask值
    #最后再根据cfg中的mask_size(28)进行resize到目标大小
    def get_target(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]#len是batch的大小，len(pos_proposals[0])才是第一幅图的pos_proposal的个数
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        #两张图像的mask_target已经cat到了一起
        #(n,28,28),n是两个图像的正proposal的个数和
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()#每一种loss都是在各自bbox_head的loss函数中定义dict(),再loss['']= ，之后再把该dict update到总的loss的dict中即可
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    """
    测试的时候为了获得最终的预测的mask，类似于bbox的get_det_bboxes
    获取的是按类分的每个预测框所在位置的mask值，当然每个mask的大小还是图片的大小，只是在预测框的位置有mask
    """
    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)] #设置语义的List,长度为80(不包含背景)
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1 #label的值0代表的是一个图片中的第一个物体所在的类，+1是为了后面从mask_pred中取出所在label的切片图

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):#对每个pro操作
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]#(28,28)
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)#设立出该幅图大小的im_mask,初始化为零，即该幅图还没有mask

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))#bbox_mask是mask_pred resize到对应的pro的大小
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)#如果预测的值>0.5就是1，否则为0
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask #在对应bbox的位置，赋值为bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]#在show_result的时候会decode 是pycocotool的函数
            cls_segms[label - 1].append(rle)#在这里把mask放在对应的label处

        return cls_segms
