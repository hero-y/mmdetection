# print(object_class)

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
#             if label['w'] < 1024 and label['h'] < 1024:
#                 ws.append(label['w'])
#             if label['h'] < 2048:
#                 hs.append(label['h'])
# ws = np.array(ws)
# hs = np.array(hs)
# plt.figure("train_data_ratio")
# ax = plt.gca()
# ax.set_xlabel('w')
# ax.set_ylabel('h')
# ax.scatter(ws, hs, c='r', s=20, alpha=0.5)
# plt.savefig("train_data_ratio")
# plt.show()

# ratios = np.array(ratios) 
# plt.hist(ratios, bins=40, density=0,facecolor="blue", edgecolor="black", alpha=0.7)
# plt.xlim((0,3))
# plt.xticks(np.arange(0,3,0.1))
# plt.tick_params(labelsize=7)
# plt.savefig("train_data_ratio")

# for data in os.listdir(test_data_path):
#     if data.endswith(".json"):
#         json_datas = os.path.join(test_data_path, data)
#         with open(json_datas,'r') as f:
#             labels = json.load(f)
#         for label in labels:
#             if label['w'] <800 or label['h'] <800:
#                 cnt +=1
# print("data:",cnt)

# test_cancer_cr34 = '/home/admin/jupyter/results/test_cancer_cr34/'
# test_cancer_cr34_data = os.listdir(test_cancer_cr34)

# with open(test_cancer_cr34 + '1502.json', 'r') as f:
#     result =  json.load(f)
# print(len(test_cancer_cr34_data))
# print(result)

# areas = []
# for data in os.listdir(train_data_split_annotations):
#     json_datas = os.path.join(train_data_split_annotations, data)
#     with open(json_datas,'r') as f:
#         labels = json.load(f)
#     for label in labels:
#         if label['class'] in ['Trichomonas']:
#             area = np.sqrt(label['w']*label['h'])
#             areas.append(area)

# areas = np.array(areas) 
# plt.hist(areas, bins=40, density=0,facecolor="blue", edgecolor="black", alpha=0.7)
# plt.xlim((0,100))
# plt.xticks(np.arange(0,100,10))
# plt.tick_params(labelsize=7)
# plt.title("Trichomonas",fontsize=20)
# plt.savefig("Trichomonas")

# areas = []
# for data in os.listdir(crop_data_annotation_tmp):
#     json_datas = os.path.join(crop_data_annotation_tmp, data)
#     with open(json_datas,'r') as f:
#         labels = json.load(f)
#     for label in labels:
#         area = np.sqrt((label['y2']-label['y1'])*(label['x2']-label['x1']))
#         areas.append(area)

# areas = np.array(areas) 
# plt.hist(areas, bins=40, density=0,facecolor="blue", edgecolor="black", alpha=0.7)
# # plt.xlim((0,200))
# # plt.xticks(np.arange(0,200,10))
# plt.tick_params(labelsize=7)
# plt.title("crop_data_area",fontsize=20)
# plt.savefig("crop_data_area")