import os
import os.path as osp
import json
import numpy as np
import glob

"""
COCO数据集整体是一个dict,然后里面有images,annotations和categories这三个关键的key,每个key中又是一个list,每个list里面有事一个个dict
所以在构建的时候，就需要先构建以个dict,在去构建里面的三个key,把三个key的内容求出来在赋值就好了

"""
classname_to_id = {"ASC-H": 1, "ASC-US":2, "HSIL":3, "LSIL":4, "Candida":5, "Trichomonas":6}

class Cancer2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
    
    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1) 

    def to_coco(self, json_path_list):
        self._init_categories()
        # cnt = 0
        for json_path in json_path_list:
            # cnt += 1
            objs = self.read_jsonfile(json_path)
            # print(objs)
            self.images.append(self._image(objs, json_path))
            # print(self.images)
            for obj in objs:
                annotation = self._annotation(obj)
                self.annotations.append(annotation)
                self.ann_id += 1
            # print(self.annotations)
            self.img_id += 1
            # if cnt == 2: 
            #     break
        instance = {}
        instance['info'] = 'cancer dataset'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

      # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)
    
     # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        image['height'] = 1536
        image['width'] = 1536
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".png")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, obj):
        label = obj['class'] 
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(0).tolist()]
        annotation['bbox'] = self._get_box(obj)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    
    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self,obj):
        return [obj['x1'], obj['y1'], (obj['x2']-obj['x1']), (obj['y2']-obj['y1'])]
        # return [obj[0], obj[1], (obj[2]-obj[0]), (obj[3]-obj[1])]



if __name__ == "__main__":

    # cancer_path = "/home/hero-y/crops/annotation_tmp/"
    # saved_coco_path ="/home/hero-y/crops/annotations/"

    cancer_path = "/home/admin/jupyter/crops/annotation_tmp/"
    saved_coco_path ="/home/admin/jupyter/crops/annotations/"
    
    json_names = os.listdir(cancer_path)
    json_path_list = [osp.join(cancer_path, json_name) for json_name in json_names]

    train_json = Cancer2CoCo()
    train_instance = train_json.to_coco(json_path_list)
    train_json.save_coco_json(train_instance, saved_coco_path + 'train.json')
