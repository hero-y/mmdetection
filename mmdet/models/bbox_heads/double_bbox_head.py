import torch.nn as nn
from mmcv.cnn.weight_init import normal_init, xavier_init

from .bbox_head import BBoxHead
from ..backbones.resnet import Bottleneck
from ..registry import HEADS
from ..utils import ConvModule

"""
里面有两个类，BasicResBlock和DoubleConvFCBBoxHead
BasicResBlock就一个__init__和一个forward，__init__就定义了三个卷积属性，前两个是3*3+1*1,第三个是为了跳连接

DoubleConvFCBBoxHead中有5个函数
初始化函数：把BasicResBlock的实例作为属性，把函数_add_conv_branch和_add_fc_branch传到属性
定义self.fc_reg和self.fc_cls，也就是forward用的基本都要在__init__定义完
在forward中：
对于卷积：先BasicResBlock(做一下缓冲),再用self.conv_branch(4个Bottleneck),最后再average,再view,再fc
对于全连接:先view,再self.fc_branch(两个fc),每个fc后加一个relu,再fc输出
"""
class BasicResBlock(nn.Module):  #3*3conv+bn+relu 1*1conv+bn 相加后再relu
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule( #里面包含了conv，norm和activate
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule( #在resnet中1*1conv后不加relu，是因为相加之后再relu
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            activation=None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out


@HEADS.register_module
class DoubleConvFCBBoxHead(BBoxHead):
    """Bbox head used in Double-Head R-CNN

                                      /-> cls
                  /-> shared convs ->
                                      \-> reg
    roi features
                                      /-> cls
                  \-> shared fc    ->
                                      \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHead, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg) #4个covn后(其实是4个bottleneck)，再avg,再fc输出坐标，每个类别都有，每张roi都输出所有类别×4的概率

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)#两个fc后再通过一个fc输出类别，每张roi都输出所有类别的概率
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self):  #利用nn.ModuleList,建立4个Bottleneck
        """Add the conv branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):  #for i in range的方法进行循环
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,  #//代表向下取整，/输出就是float
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self): #利用nn.ModuleList,建立2个fc
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels * self.roi_feat_size *
                self.roi_feat_size if i == 0 else self.fc_out_channels) #第一个fc的输入是self.in_channels * self.roi_feat_size *self.roi_feat_size
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch: #用for in 对nn.ModuleList取值
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)  #nn.AvgPool2d(roi_feat_size)

        x_conv = x_conv.view(x_conv.size(0), -1)  #torch的图像格式是(N*C*W*H),通过语句转换成(N,CWH)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1) #先view成(n,CWH)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc)) #fc+relu

        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred  #(n,m),(n,m*4)  n是roi的个数，m是类别数
