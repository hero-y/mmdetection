import torch

from mmdet.ops.nms import nms_wrapper

"""
一个函数multiclass_nms,在test中的rcnn后使用，rpn不用多类的nms,因为rpn的cls或bbox的输出都没有类别信息
在test的时候也就一个图像
faster rcnn的multi_bboxes是(n,class*4),cascade_rcnn的multi_bboxes是(n,4)
首先写出使用的nms的方法，接着对每个类遍历，for i in range(类长)，每次遍历中：先根据score的阈值筛选掉一些，
再把bbox和score拼接到一起，用nms,因为nms需要用到score,是从大分数向下筛，每次遍历生成label,和bbox对应起来
把每个类选出的bbox放到list中，再cat，最后再sort排序，选出前max_num个[:max_num]
"""
def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')#pop移除列表中的某个元素，并返回该元素的值
    nms_op = getattr(nms_wrapper, nms_type)
    #函数名字叫multiclass_nms，所以是按照类别循环的,所以对于分数而言，就要和每一类对应上去,
    #首先确定每个对象的该类的分数是否大于阈值，做一次过滤,再把对应该类的分数和score_factors相乘,
    #score_factors在fcos中是centerness,把bboxes和scores cat到一起，输入nms中，输出的cls_dets也是5位的
    #可能nms后，bbox的数量会超过设定的数量,所以再根据分数排序
    for i in range(1, num_classes): #在coco中因为每个bbox的score都是81个类这么多,要按照类来操作
        cls_inds = multi_scores[:, i] > score_thr  #获取该类的满足阈值的所有bbox的序号,0,1分布
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if multi_bboxes.shape[1] == 4:
            _bboxes = multi_bboxes[cls_inds, :]
        else:
            _bboxes = multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)  #先把bbox和score拼接到一起,前四个是坐标,最后一个是分数score用[:,None]增维
        cls_dets, _ = nms_op(cls_dets, **nms_cfg_)  #用nms,nms需要使用分数信息
        #注意下面生成label的方式:因为每个位置会预测80个置信分数，并不是说只取最大的置信分数对应的类作为bbox的类
        #因为score_thr比较小，所以对每一个类都进行一次判断是不是大于score,即一个bbox可能会有多个类出现
        cls_labels = multi_bboxes.new_full((cls_dets.shape[0], ),
                                           i - 1,
                                           dtype=torch.long)  #把bbox和label对应起来
        bboxes.append(cls_dets)
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num: #在test的时候每次只检测一张图像,所以返回的bboxes不带有图像id信息,就不用考虑该bbox应该在哪张图片上画框
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
