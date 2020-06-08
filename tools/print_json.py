import json

minicoco_json_path ='data/coco/annotations/minicoco_train2017.json'


if __name__ == '__main__':
    with open(minicoco_json_path ) as f:
        minicoco_json = json.load(f)
        print(minicoco_json)