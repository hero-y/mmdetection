import random
import os
import os.path as osp
from  to_coco import Cancer2CoCo
import glob
import shutil

# crop_data_root = '/home/admin/jupyter/crops'
crop_data_root = '/home/hero-y/crops' 

data_path = os.path.join(crop_data_root,'train')
annotation_tmp_path = os.path.join(crop_data_root,'annotation_tmp')  

val_data_path = os.path.join(crop_data_root,'val')  
val_annotation_tmp_path = os.path.join(crop_data_root,'val_annotation_tmp') 
val_annotations = os.path.join(crop_data_root,'val_annotations')  

def gen_val_datasets(data_path, annotation_tmp_path, val_data_path, val_annotation_tmp_path, val_annotations, ratio = 0.1):
    data_files = os.listdir(data_path)
    val_files = random.sample(data_files,int(len(data_files)*ratio))
    for file in val_files:
        shutil.move(data_path + "/" + file,val_data_path + "/" + file)
        shutil.move(annotation_tmp_path + "/" + file.replace('.png','.json'),val_annotation_tmp_path + "/" + file.replace('.png','.json'))
   
    json_names = os.listdir(val_annotation_tmp_path)
    json_path_list = [osp.join(val_annotation_tmp_path, json_name) for json_name in json_names]

    train_json = Cancer2CoCo()
    train_instance = train_json.to_coco(json_path_list)
    train_json.save_coco_json(train_instance, val_annotations + '/' + 'val.json')

if __name__ == "__main__":
    gen_val_datasets(data_path, annotation_tmp_path, val_data_path, val_annotation_tmp_path, val_annotations)
