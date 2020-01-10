import os

train_data_root = '/home/admin/jupyter/Data/train/'
train_data_split_root = '/home/admin/jupyter/train_data_split'
train_data_split_pos_root = '/home/admin/jupyter/train_data_split/pos'
train_data_split_neg_root = '/home/admin/jupyter/train_data_split/neg'
train_data_split_annotations = '/home/admin/jupyter/train_data_split/annotations'

# train_data_root = '/home/hero-y/Data/train/'
# train_data_split_root = '/home/hero-y/train_data_split'
# train_data_split_pos_root = '/home/hero-y/train_data_split/pos/'
# train_data_split_neg_root = '/home/hero-y/train_data_split/neg/'
# train_data_split_annotations = '/home/hero-y/train_data_split/annotations/'

if __name__ == '__main__':
    datas_all = os.listdir(train_data_root)
    annotations = []
    for data in datas_all:
        if data.endswith(".json"):
            os.symlink(os.path.join(train_data_root,data),os.path.join(train_data_split_annotations,data))
            annotations.append(data) 
    for data in datas_all:
        if data.endswith(".kfb"):
            if data.replace('.kfb','.json') in annotations:
                os.symlink(os.path.join(train_data_root,data),os.path.join(train_data_split_pos_root,data))
            else:
                os.symlink(os.path.join(train_data_root,data),os.path.join(train_data_split_neg_root,data))