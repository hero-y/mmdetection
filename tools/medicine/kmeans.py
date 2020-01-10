import numpy as np
import json
import os
import os.path as osp
import random
import matplotlib.pyplot as plt

annotation_path = '../../data/cancer/crops/annotation_tmp'

def read_jsonfile(path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:     #box(c_w,c_h)完全包含box(w,h)
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:   #box(c_w,c_h)宽而扁平
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities)


def kmeans(X,centroids,eps):
    
    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)    
    iter = 0
    old_D = np.zeros((N,k)) #距离矩阵  N个点,每个点到k个质心 共计N*K个距离

    while True:
        D = []#保留每个点到k个质心的iou距离 (N,K)
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)  #d是一个k维的   
            D.append(d)   
        D = np.array(D) # D.shape = (N,k)
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1) #返回每一行的最小值的下标.即当前样本应该归为k个质心中的哪一个质心.
        
        if (assignments == prev_assignments).all() :  #质心已经不再变化
            print("Centroids = ",centroids)
            return

        #calculate new centroids   
        centroid_sums=np.zeros((k,dim),np.float)  #(k,2)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]        #将每一个样本划分到对应质心
        for j in range(k):            
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j)) #更新质心
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  

if __name__ == "__main__":
    eps = 0.005
    num_clusters = 5
    json_names = os.listdir(annotation_path)
    json_path_lists = [osp.join(annotation_path, json_name) for json_name in json_names]
    ws = []
    hs = []
    annotation_dims = []
    for json_path_list in json_path_lists:
        objs = read_jsonfile(json_path_list)
        for obj in objs:
            w = obj[2] - obj[0]
            h = obj[3] - obj[1]
            ws.append(w)
            hs.append(h)
            annotation_dims.append(tuple(map(float,(w,h))))
    annotation_dims = np.array(annotation_dims)
    ws = np.array(ws)
    hs = np.array(hs)
    print("load OK")
    """
    plt画图
    """
    plt.figure("scatter fig")
    ax = plt.gca()
    ax.set_xlabel('w')
    ax.set_ylabel('h')
    ax.scatter(ws, hs, c='r', s=20, alpha=0.5)
    plt.show()
    """
    Kmeans聚类
    """
    indices = [ random.randrange(annotation_dims.shape[0]) for i in range(num_clusters)] #随机选取序号
    centroids = annotation_dims[indices]
    print("init_centroids=",centroids)
    kmeans(annotation_dims,centroids,eps)

    #绘制比例图
    # train_data_split_annotations = '/media/hero-y/机械盘T1/Tianchi/medicine/annotations'
    # ws=[]
    # hs=[]
    # ratios = []
    # for data in os.listdir(train_data_split_annotations):
    #     json_datas = os.path.join(train_data_split_annotations, data)
    #     with open(json_datas,'r') as f:
    #         labels = json.load(f)
    #     for label in labels:
    #         if label['class'] not in ['roi']:
    #             ratio = label['h']/label['w']
    #             ratios.append(ratio)
                
    # ratios = np.array(ratios)
    # # plt.figure("train_data_ratio")
    # plt.hist(ratios, bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.xlim((0,3))
    # plt.xticks(np.arange(0,3,0.1))
    # plt.show()

