import mmcv
import numpy as np
import torch
from torch.nn.modules.utils import _pair


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]#len为2，即图像的batch数
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = torch.cat(list(mask_targets))
    return mask_targets

#按图像输入，把正样本的坐标，和对应的label都.cpu().numpy(),这样是为了mmcv.imresize,之后再变回tensor
#对每个proposal单独操作，先获得该proposal对应的mask，gt_masks[pos_assigned_gt_inds[i]]
#求出每个proposal的坐标以及h,w,gt_mask[y1:y1+h,x1:x1+w]代表该proposal在原图所在区域的mask的取值
#再对它用imresize到(28,28)即为该proposal的mask的target
#再把每个proposal的mask_target添加到一个list中,再np.stack(默认axis=0),把mask_target堆叠到一起(n,28,28)
#再转换到tensor,并.float(这个是因为loss用binary_cross_entropy_with_logits时target需要用float)
def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    mask_size = _pair(cfg.mask_size) #变成双数(28,28)
    num_pos = pos_proposals.size(0)#proposal的总数，为了对每个proposal单独操作，用.size(0)，而不是len
    mask_targets = []
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
        for i in range(num_pos):
            gt_mask = gt_masks[pos_assigned_gt_inds[i]]
            bbox = proposals_np[i, :].astype(np.int32)
            x1, y1, x2, y2 = bbox
            w = np.maximum(x2 - x1 + 1, 1)
            h = np.maximum(y2 - y1 + 1, 1)
            # mask is uint8 both before and after resizing
            # mask_size (h, w) to (w, h)
            target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w],
                                   mask_size[::-1])#(28,28)
            mask_targets.append(target)
        mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(
            pos_proposals.device)#(n,28,28),n是每个图像的正proposal的个数
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
    return mask_targets
