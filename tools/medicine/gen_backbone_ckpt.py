import argparse
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Generate backbone from detector checkpoint

    """
    checkpoint = torch.load(in_file) #导入参数权重
    # print(checkpoint['state_dict'],keys())#打印所有的state_dict的keys
    in_state_dict = checkpoint.pop('state_dict') #state_dict中的是参数的key，以及对应的参数,因为一个文件里面可能会有对个Key,除了state_dict之外
    out_state_dict = OrderedDict() #建立顺序dict
    for key, val in in_state_dict.items(): #对每个参数的dict遍历，其中包括key和val,backbone中的key都是ba
        # delete 'backbone' in keys
        if 'backbone' in key:
            new_key = key.replace('backbone.', '')#其中backbone的keys的形式如backbone.conv1.weight,所以把backbone.去掉即可
            out_state_dict[new_key] = val

    checkpoint['state_dict'] = out_state_dict
    # print(checkpoint['state_dict'].keys())
    torch.save(checkpoint['state_dict'], out_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate backbone checkpoint from detector')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)
    #python tools/medicine/gen_backbone_ckpt.py ../models/Cascade-R-CNN/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth ../models/backbone.pth

if __name__ == '__main__':
    main()
