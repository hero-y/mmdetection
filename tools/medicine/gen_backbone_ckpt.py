import argparse
from collections import OrderedDict

import torch


def convert(in_file, out_file):
    """Generate backbone from detector checkpoint

    """
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    for key, val in in_state_dict.items():
        # delete 'backbone' in keys
        if 'backbone' in key:
            new_key = key.replace('backbone.', '')
            out_state_dict[new_key] = val

    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint['state_dict'], out_file)


def main():
    parser = argparse.ArgumentParser(
        description='Generate backbone checkpoint from detector')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()
