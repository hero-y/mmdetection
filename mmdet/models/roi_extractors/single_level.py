from __future__ import division

import torch
import torch.nn as nn

from mmdet import ops
from mmdet.core import force_fp32
from ..registry import ROI_EXTRACTORS

"""
一个类SingleRoIExtractor，继承nn.Module,共7个函数
初始化函数构建roi_layers,也就是RoIPool或者RoIAlign等，输入是特征图以及rois(img_id+在原图上的坐标)
注意的是featmap_strides是[4,8,16,32]而没有64的原因，我认为是防止rois的大小刚符合32那个stride的时候，rois/64的大小会是7.0几，所以就不要64了
而finest_scale=56也是有原因的，56=7*8
- scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3
这里的scale是对面积的开方，也就可以当做是w或者h,scale<112的时候为level0(stride = 4),如果是113就是level1,stride=8,在该特征图的大小就是14.125，
14.125是7的2倍，就比较合适做maxpooling，后面依次类推
在forward中先调用map_roi_levels求target_lvls,再定义roi_feats的大小，初始化为0，考虑是否要scale,再对4个level进行迭代，
判断target_lvls == i,作为索引代入rois,并和feats[i]一同代入roi_layers，最会输出的是(n,256,7,7),n是roi的个数，这里就没有img_id，所以img_id应该也就是在RoIAlign中去对应图像的时候用的
"""
@ROI_EXTRACTORS.register_module
class SingleRoIExtractor(nn.Module):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False

    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1)) #求rois的面积，都加1是为了防止根号下出现零,从左上角右下角坐标求w,h时都要+1
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6)) #floor是向下取整，ceil是向上取整
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()  #clamp钳，做一个分类讨论，输入值小于min,则为min,中间则为自己，大于max，则为max
        #输入的rois的坐标应该是原图片上的坐标，rois越大，说明检测的物体越大，所以层级应该越高
        #target_lvls的shape是(n,),n是proposals的总个数
        return target_lvls
    
    #改变scale就是先由左上右下求中心坐标和w,h(注意+1),再对w,h进行scale,再有中心坐标和w,h变回左上右下(左上+0.5，右下-0.5)
    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1) #记得把img_id给拼接回去
        return new_rois

    @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)

        out_size = self.roi_layers[0].out_size  #7
        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i #必须是torch.tensor才行。torch.tensor([1,2,3,4]),target_lvls和i相等的对应的地方再inds中是1，否则是零
            if inds.any():
                rois_ = rois[inds, :] #取出是1的那行序号，也就是当前特征图中的roi
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t #先建立好roi_feats的形状，然后用[]对应的赋值即可，此处是相加
        return roi_feats ##最后返回的是所有roi_feats，大小为(rois.size()[0], self.out_channels,out_size, out_size)
