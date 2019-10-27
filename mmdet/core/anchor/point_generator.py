import torch


class PointGenerator(object):

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y)) #torch的repeat是一组一组的复制，np的repeat是一个一个的复制
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    #直接使用torch.arange,而不是用np,在用arange的里面加上了device='cuda',是因为最后返回的point是要和特征图的输出做运算的
    #在cuda上的变量不能和在cpu上的变量做乘加操作，但是可以和数值做操作，如stride=16,它不是torch生成的，就只是一个常量，而不是torch生成的变量
    def grid_points(self, featmap_size, stride=16, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)
        return all_points

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
        return valid
