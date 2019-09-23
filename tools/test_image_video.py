from mmdet.apis import init_detector, inference_detector, show_result, show_result_pyplot
import mmcv
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--img', help='img file')
    parser.add_argument('--video', help='video file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    if args.img is not None:
        result = inference_detector(model, args.img)
        show_result(args.img, result, model.CLASSES, score_thr=0.5)
    if args.video is not None:
        video = mmcv.VideoReader(args.video)
        for frame in video:
            result = inference_detector(model, frame)
            show_result(frame, result, model.CLASSES, score_thr=0.5, wait_time=1)


if __name__ == '__main__':
     main()
