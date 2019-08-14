import numpy as np
import torch

from .base_sampler import BaseSampler

"""
一个类，有4个函数：初始化，_sample_pos,_sample_neg,random_choice
_sample_pos使用了torch.nonzero(assign_result.gt_inds>0)就直接把pos_inds取出来了
再防止超过了规定的个数，使用random_choice去随机取，其实也就是把这么多数变成乱序，取出约定的个数，再把这些作为序号传进去即可
"""
class RandomSampler(BaseSampler):

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)

    @staticmethod
    def random_choice(gallery, num): #先arange
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num  #len:一维时是总数量，二维或多维都是第一维的数量，numel()是所有个数
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery)) #arange和range的区别:arange需要torch或np,返回的是tensor或ndarray，range返回的是List,在这里面需要用arange因为下位用np.random.shuffle.要对ndarray类型使用
        np.random.shuffle(cands) #return None，是直接对cands乱序，原cands已经不在
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device) #from_numpy .to()
        return gallery[rand_inds]

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0)  #torch.nonzero返回的是(m,n),m是不为零的个数，n是输入的维度
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
