import torch.nn as nn

"""
一个类Accuracy，继承nn.Module,一个初始化函数传入topk,一个accuracy,参数是pred和target
accuracy中pred是对每个类都有预测，首先pred.topk，值不重要，把序号取出来，如果topk=5,pred_label是(n,5)
再转置变成(5,n)为了target方便，target也expand_as为(5,n),再.eq相等的就是1了，再用for对topk迭代，
correct[:k]，把前k行的取出，sum求总个数，再×100/总的rois的数量
"""
def accuracy(pred, target, topk=1):
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk) #取出最大的，如果max(1,5),maxk是5
    _, pred_label = pred.topk(maxk, dim=1) #如果maxk=5,pred_label是(n,5)，这里面的5个值也是根据value排序后的
    pred_label = pred_label.t()  #.t()其实也就是transpose，即更换维度
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label)) #.eq是判断是否相等，.expand_as是复制输入的值，让他的维度和目标值相等
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  #correct[:k]是因为要求出不同k值的acc,所以用这个方式去除符合的correct
        #.view(-1)是转换成1维，sum，是把所有的相加，即所有correct的数值
        res.append(correct_k.mul_(100.0 / pred.size(0))) #.mul就是乘法的操作
    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return accuracy(pred, target, self.topk)
