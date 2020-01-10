import os
import os.path as osp
import json

# result_file1_path = "/home/admin/jupyter/results/test_cancer_cr34_fixtest2/"
# result_file2_path = "/home/admin/jupyter/results/test_cancer_cr34_small/"
# save_file_path = "/home/admin/jupyter/results/test_cancer_cr34_fixtest2_small_merge/"

result_file1_path = "/home/hero-y/results/test_cancer_cr34_fixtest2/"
result_file2_path = "/home/hero-y/results/test_cancer_cr34_small/"
save_file_path = "/home/hero-y/results/test_cancer_cr34_fixtest2_small_merge/"

def main():
    result_file1_datas = os.listdir(result_file1_path)
    result_file2_datas = os.listdir(result_file2_path)
    assert len(result_file1_datas) == len(result_file2_datas), ('files is not matched')
    for data in result_file1_datas:
        merge_data = []
        save_path = osp.join(save_file_path,data)

        data1_path = osp.join(result_file1_path,data)
        with open(data1_path, 'r') as f:
            data1s = json.load(f)
        for data1 in data1s:
            if data1['class'] in ['ASC-H','ASC-US','HSIL','LSIL','Candida']:
                merge_data.append(data1)
        
        data2_path = osp.join(result_file2_path,data)
        with open(data2_path, 'r') as f:
            data2s = json.load(f)
        for data2 in data2s:
            if data2['class'] in ['Trichomonas']:
                merge_data.append(data2)
        
        with open(save_path, 'w') as f:
                json.dump(merge_data, f)
    
    print("save_data_size:",len(os.listdir(save_file_path)))

if __name__ == '__main__':
     main()
