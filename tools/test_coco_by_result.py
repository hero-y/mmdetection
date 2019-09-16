import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
求出在各个阈值下的AP值，把下面的代码粘提到cocoEval.summarize()中的def _summarizeDets()的对应位置即可
其中iou=0.9,算出来的阈值为-1，比较奇怪，画图的时候跳过改点
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.55, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.6, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, iouThr=.65, maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=.7, maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[7] = _summarize(1, iouThr=.8, maxDets=self.params.maxDets[2])
            stats[8] = _summarize(1, iouThr=.85, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(1, iouThr=.90, maxDets=self.params.maxDets[2])
            stats[10] = _summarize(1, iouThr=.95, maxDets=self.params.maxDets[2])
"""
def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='config')
    parser.add_argument('resultfile', help='resultfile')
    parser.add_argument(
        'eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    dataset = build_dataset(cfg.data.test)

    result_files = args.resultfile
    eval_types = args.eval

    coco = dataset.coco
    coco_dets = coco.loadRes(result_files)
    img_ids = coco.getImgIds()
    for res_type in eval_types:
        iou_type = 'bbox' if res_type == 'proposal' else res_type 
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

if __name__ == '__main__':
    main()
