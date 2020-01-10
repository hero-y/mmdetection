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


def transfer_datasets(data_path, annotation_tmp_path, val_data_path, val_annotation_tmp_path):
    data_files = os.listdir(val_data_path)
    for file in data_files:
        shutil.move(val_data_path + "/" + file, data_path + "/" + file)
        shutil.move(val_annotation_tmp_path + "/" + file.replace('.png','.json'), annotation_tmp_path + "/" + file.replace('.png','.json'))

if __name__ == "__main__":
    transfer_datasets(data_path, annotation_tmp_path, val_data_path, val_annotation_tmp_path)
