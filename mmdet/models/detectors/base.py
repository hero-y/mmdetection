import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch.nn as nn

from mmdet.core import auto_fp16, get_classes, tensor2imgs


class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    #下面的show_result只会是在show的时候才会进入,也就是不会rescale
    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset=None,
                    score_thr=0.3):
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        img_tensor = data['img'][0] #取出第一个图像(是tensor类型)，其实在test的时候一次也就一个图片
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg) #图像通过网络前要预处理就变成了tensor(在custom中),该函数就是把tensor重新变换维度,变成在cpu上,numpy格式，去归一化
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            bboxes = np.vstack(bbox_result)  #垂直的把数组堆叠起来(n,5)，这是numpy堆叠数组的方式和torch不一样
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result) #segm_result是按类的list,每个类里面也是一个list,concat_list就是把这些list,变成一个list
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:#对每一个pro对应的mask操作，直接改变要展示的图片上的mask位置的rgb值
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)#(1,3)代表一行3列，即rgb
                    mask = maskUtils.decode(segms[i]).astype(np.bool)#解码出的是mask的位置
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5 #该mask位置的颜色值为原颜色值×0.5+随机生成的颜色值×0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ] #bbox_result仍然是按类的list,bboxes才是vstack后的
            labels = np.concatenate(labels)
            mmcv.imshow_det_bboxes( #输入图像img,bboxes(有score),labels,class_name,score_thr
                img_show,
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                bbox_color='blue',
                text_color='white',
                thickness=2
                )
