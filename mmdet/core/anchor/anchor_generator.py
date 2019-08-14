import torch
#该文件只有一个class
#主要包含函数：1.生成基本ahchor的函数2.grid_anchors按照feat在每个位置生成anchor的函数3.valid_flags按照feat在每个位置生成valid的函数
class AnchorGenerator(object): 

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)  #torch.tensor生成的数据类型就是输入的数据类型，而torch.Tensor输出的是Float型。Long型就是int64,int32
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors() #这些init再导入数据前就完成了

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios) #根号
        w_ratios = 1 / h_ratios #1/根号
        #使用w_ratios[:None]的原因是self.scales中可能有多个值，如果有多个值，二者就不能相乘，就要转化为矩阵所以[:,None]和[None,:]
        #w_ratios[:, None]为(3,1)加了一个None相当于是多增加了一维
        #view(-1)是把tensor转化为一维的tensor
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1) 
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)
        # yapf: disable
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),  #x_ctr是普通值，也可以加减乘除tensor值
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round() #返回浮点数的四舍五入值 shpe是(3,4)
        # yapf: enable
        return base_anchors

    def _meshgrid(self, x, y, row_major=True): #网格点坐标矩阵 
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device) #(3,4)
        #卷积过后的特征图上的每一个位置都会有3个anchor，一共的位置数就是特征图的h*w
        #对于该特征图而言，anchor其实是原图上体现的，所以就要根据base_anchor(anchor的起始位置)加上shift偏移量来算其余anchor的坐标
        #self._meshgrid生成的shift_xx和shift_yy的shape都是(len(shift_x)*len(shift_y),),而len(shift_x)=w,len(shift_y)=h
        #shift_xx形如：([0,1,2,3,0,1,2,3...])
        #shift_yy形如：([0,0,0,1,1,1,2,2,2...])
        #这样的目的起始就是base_anchor加shift时，先y固定，x变化，自左向右，自上向下的加
        #torch.stack生成的shifts的形状为(len(shift_x)*len(shift_y),4) len(shift_x)*len(shift_y)个位置，都有4个变化点
        #shifts形如:([ [0,0,0,0],
        #             [4,0,4,0],
        #             [8,0,8,0],
        #             [0,4,0,4],
        #             [4,4,4,4], 
        #             [8,4,8,4]
        #             .........
        # ]])
        #base_anchors[None, :, :]扩充一个维度，该维度代表有几个放anchor的位置，大小为len(shift_x)*len(shift_y)
        #shifts[:, None, :]扩充一个维度，代表每个位置，有几个anchor,3个
        #base_anchors[None, :, :]+shifts[:, None, :]加的方法就是想扩张维度再加
        feat_h, feat_w = featmap_size  #(c,h,w)
        shift_x = torch.arange(0, feat_w, device=device) * stride  #这个stride其实也是该阶段的特征图相对于原图来说缩放的大小
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)  #变成float
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)  
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        #在使用torch.arange生成shift_x的时候,因为用了torch和device='cuda',之后的基于该量操作生成的变量都是tensor，且在cuda上
        #cuda只能和cuda上的数据操作，所以base_anchors.to(device),变到了cuda上
        return all_anchors  #(K*A, 4)这个是一个特征图的anchor,5个特征图的anchor被分开放到一个List中

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1) #在view前有的时候需要使用contiguous，把数据变得连续
        #valid[:,None],加了一个None就可以当做是valid[:,1],如果valid是[1,1],则valid[:,None]是[[1],[1]],就从(2,)变成了(2,1)
        return valid  #(feat_w*feat_h*3,)
