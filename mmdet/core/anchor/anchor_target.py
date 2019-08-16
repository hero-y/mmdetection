import torch

from ..bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner
from ..utils import multi_apply

"""
在anchor_target文件中没有类，有5个函数
首先是anchor_target,该函数的功能是：首先把anchor_list按图像list的方式，把所有层的放到一起
在使用multi_apply函数对每幅图像单独操作，这样的原因是预处理的时候每个data里都是dict,而每个dict又是按图像顺序来放每一个值的
调用了anchor_target_single，生成了按图像的list，有all_labels,all_label_weights,all_bbox_targets,all_bbox_weights,这是每一个anchor都会有的
之后再用images_to_levels,把排序方式变成了level,具体流程是把输入值用torch.stack按图像拼接，用for in 知道每个level中的anchor数量，在通过切片的方式把对应的数量放到List中

再讲anchor_target_single,首先用anchor_inside_flags，进一步把flag缩小，带入索引后，生成了新的valid_anchor,
对这些anchor,采用assign_and_sample,要注意sample中的几个量，sample.pos_inds,代表正anchor的序号，pos_bboxes,代表正的bbox的值，pos_gt_bboxes,代表正的bbox对应的gt的坐标值
设定bbox_targets,bbox_weights,labels,label_weights,这些的第一维的大小都是有效anchor的数量，最后会通过unmap转换成所有anchor的数量
再用bbox2delta,输入pos_bboxes和pos_gt_bboxes求出目标值，再把pos_inds序号带入bbox_targets中，且把刚才求出的目标值带入，这样就对应起来了
把pos_inds这部分序号的bbox_weights设为1，其余默认为零，所以对于bbox来说，在loss的时候只有正样本有用
而label也是按序号带入正值，正值是gt_labels[sampling_result.pos_assigned_gt_inds]，把pos被安排的gt的序号带入gt_labels中，gt_labels是从json中得到的，从sample中得到的前面一般会加pos或neg
,不同的是，label_weights的neg_inds也会被赋值成1.0，所以在loss的时候label的负样本也有用，再通过Unmap，把序号和样本值再次对应求出在原anchor的值
"""
def anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]] #anchor_list是list[list] anchor_list[0]里面还是一个list(这个0代表的是batch中的第一幅图片的anchorlist,一个batch的图像尺寸是一样的，所以anchor_list也是一样的)，装着5个level的anchor,每个level的shape是(K*3,4),K是该特征图的h*w
    #num_level_anchors是个list,list没有cuda,其中的每一个值，也不在cuda上，因为.size(0)
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])  #把所有阶段的anchor放到一起，因为是要在原图上去求target,但图像和图像之间还是用list区分的
        valid_flag_list[i] = torch.cat(valid_flag_list[i]) #把所有阶段的valid_flag放到一起，因为是要在原图上去求target

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    #multi_apply获得的值都是个List,all_labels就是个List,里面的迭代对象是图像的序号
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list]) #sum是对一个list里面的值求和，所以sum([]),[]里面经常是用for in产生的，所以for in 也在[]里面
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)  #num_level_anchors:每个level的anchor总数量
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors) #按照层级进行list是因为预测的cls_scores和bbox_preds也是按照层级来的list,为了便于loss
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    #target是个List,0代表0维，若两幅图，原target是[label,label],只能对两幅图的label单独操作
    #使用torch.stack(target, 0)后，变成(2,len(label)),这时就可以用target[:, start:end]对维度中的序号进行操作了
    target = torch.stack(target, 0) 
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0)) #squeeze(0)是，如果就一副图像，就用squeeze(0)把第一维去掉
        start = end
    return level_targets #该list中共5个对象，因为有5个level层，每个对象的shape为(2,n),n是每个层的anchor数


def anchor_target_single(flat_anchors,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)  #img_meta['img_shape'][:2]:(h,w,C)变成(h,w)
    if not inside_flags.any(): #.any()判断一个迭代参数是否全都是Fasle,有一个True则返回的就是True
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]  #只把有效flag的anchor取出来，来assign和sample等，无效的根本不用管
    #inside_flags在cuda上
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]  #这里的anchor已经是有效的anchor了，就是用valid_flag过滤了一下的
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    #用torch.zeros_like和anchors.new_zeros()生成的bbox_targets，bbox_weights，labels，label_weights都是在cuda上
    pos_inds = sampling_result.pos_inds  #正anchor的序号
    neg_inds = sampling_result.neg_inds
    #sampling_result取出来的也都在cuda上
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means, target_stds)  #anchor的target就是assign和sample之后的gt，再用bbox2delta取转化，和rcnn中一样
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]  #求label,也就是把anchor经过assign后的gt的序号带入gt_labels中即可
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    #因为前面用flat_anchors[inside_flags, :]把一些anchor滤出掉了，用滤出的anchor求的label，bbox,
    #这样可能是为了减少一些干扰，之后再把得到的label,bbox映回到原大小的anchor中
    #使用unmap函数，输入现值，anchor总数量(未经过valid_flag滤出的)，inside_flags
    if unmap_outputs:  
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds) #labels:(num_total_anchors,)总anchor数为原图中全部的anchor,即5个特征图的长*宽之和*3

#valid_flags:(K1*A+K2*A+..K5*A,) flat_anchors:(K1*A+K2*A+..K5*A,4)
#valid_flags和flat_anchors的形状的第一维是相同的，都是5层特征图的面积和×3
#这个函数就是为了用flat_anchors的四个坐标再次对valid_flags做一个&，一个限定，让anchor不会超出边界
def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2] #img_shape是(h,w),是一个元组，通过[:2]也是可以取的
    #img_shape本就是个tuple,没有cuda
    if allowed_border >= 0: 
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1: #进入
        ret = data.new_full((count, ), fill) 
        ret[inds] = data #ret的大小还是count(即num_total_anchors)，用inds后，就提取出对应的位置，赋值data,没提取的默认为0
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret
