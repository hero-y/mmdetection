import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS

#主要就是该文件+CustomDataset+单独的pythoncocoapi
"""
一个类CocoDataset,继承了CustomDataset，有CLASSES和4个函数
首先是load_annotations函数，该函数返回的是img_infos,即把json文件用cocoapi导入进来，并把Img字段放在img_infos中
第二个是get_ann_info，输入图像的序号，获取该id，然后把该张图的所有相关的ann字段获取到，
是个list,返回的是对这个List中的多个ann字段解析
即_parse_ann_info函数，解析出gt_bbox和gt_label和gt_bboxes_ignore值，分别都是List,然后
再把这些放到dict中,建立键值dict()
最后是_filter_imgs就是把一些img字段中长宽太小的图像滤出掉
"""
@DATASETS.register_module
class CocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)  #self.cat_ids没有指定参数时，返回的是所有类的id,建立一个dict,键值是类id,值是Label
        }
        self.img_ids = self.coco.getImgIds() #self.img_ids是个List
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0] #获取该id的图像字段
            info['filename'] = info['file_name']  #原先没有filename这个键值
            img_infos.append(info)  
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    #ann中有ann+img+category等类，每个里面都有多个分类
    #Img主要参数是img_id,h,w
    #ann的主要参数是bbox,img_id,cat_id,id
    #cat的主要参数是cat_id
    #一个ann代表的是一个Bbox的主要参数，所以一个img可以有多个ann，可以通过ann的img_id来知道在那个id中，通过cat_id，知道该Bbox是哪个类
    #返回一个dict
    def _parse_ann_info(self, img_info, ann_info):
        
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            #bbox在annotation的文件中是(x1,y1,w,h)即左上角坐标和宽高值，在提取每个batch的时候就转变成左上和右下坐标了
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]#根据左上角坐标和宽高求右下角坐标时，要减1
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])#根据对应的ann中的category_id，输入到cat2label中求label,cat2label把类别数+1作为label，因为label=0是背景
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
