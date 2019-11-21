from mmdet.apis import init_detector, inference_detector, show_result, show_result_pyplot
import mmcv
import argparse
import time
import kfbReader
import numpy as np
import os
import os.path as osp
import json

result_file = '/home/hero-y/tianchi'
result_file_new = '/home/hero-y/tianchi_new'

def read_jsonfile(path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

def main():
    result_names = os.listdir(result_file)
    for result_name in result_names:
        print(result_name)
        result_data_path = osp.join(result_file, result_name)
        save_path = osp.join(result_file_new, result_name)
        result_datas = read_jsonfile(result_data_path)
        save_data = []
        for result_data in result_datas:
            # print(result_data)
            # result_data['x'] = (np.floor(result_data['x']/1024)*800)+(result_data['x']%1024)
            # result_data['y'] = (np.floor(result_data['y']/1024)*800)+(result_data['y']%1024)
            # print(result_data)
            roi_save_data = {}
            roi_save_data["x"] = (np.floor(result_data['x']/1024)*800)+(result_data['x']%1024)
            roi_save_data["y"] = (np.floor(result_data['y']/1024)*800)+(result_data['y']%1024)
            roi_save_data["w"] = result_data['w']
            roi_save_data["h"] = result_data['h']
            roi_save_data["p"] = result_data['p']
            save_data.append(roi_save_data)
            # print(save_data)
        with open(save_path, 'w') as f:
            json.dump(save_data, f)
            # print("OK")
            # print(save_data)
            # break
        # print(len(result_datas))
        # print(result_data_path)
        # break
    # print(len(test_names))

def main2():
    result_names = os.listdir(result_file)
    for result_name in result_names:
        print(result_name)
        result_data_path = osp.join(result_file, result_name)
        save_path = osp.join(result_file_new, result_name)
        result_datas = read_jsonfile(result_data_path)
        save_datas = read_jsonfile(save_path)
        print(result_data_path)
        print(save_path)
        for result_data in result_datas:
            print(result_data)
            break
        for save_data in save_datas:
            print(save_data)
            break
        break
        # for result_data in result_datas:
        #     # print(result_data)
        #     # result_data['x'] = (np.floor(result_data['x']/1024)*800)+(result_data['x']%1024)
        #     # result_data['y'] = (np.floor(result_data['y']/1024)*800)+(result_data['y']%1024)
        #     # print(result_data)
        #     roi_save_data = {}
        #     roi_save_data["x"] = (np.floor(result_data['x']/1024)*800)+(result_data['x']%1024)
        #     roi_save_data["y"] = (np.floor(result_data['y']/1024)*800)+(result_data['y']%1024)
        #     roi_save_data["w"] = result_data['w']
        #     roi_save_data["h"] = result_data['h']
        #     roi_save_data["p"] = result_data['p']
        #     save_data.append(roi_save_data)
        #     # print(save_data)

if __name__ == '__main__':
     main()
