import pandas as pd
from default_config import *
import numpy as np

path = root_path + r'\log\test\ShuffleNetv2\DIFGSM\ResUnet/'
name = r'ResNet18_PGD_2022_07_05_17.log'
f = np.array(open(path + name).readlines())
mode_index = np.array([3,4,5,6,7])
result_list = []
dict = {}

for i in range(int(f.size/2)):
    info = np.array(f[2*i].strip().split(':'))[mode_index]
    data = np.array(f[2*i+1].strip().split(':'))[mode_index]
    result_list.append((info,data))

for (info,data) in result_list:
    index0 = info[0].split(' ')[0]
    index1 = info[2].split(' ')[0]
    index2 = info[3].split(' ')[0]
    index3 = info[4].split(' ')[0]
    if index0 not in dict:
        dict[index0] = {}
    if index1 not in dict[index0]:
        dict[index0][index1] = {}
    if index2 not in dict[index0][index1]:
        dict[index0][index1][index2] = {}
    if index3 not in dict[index0][index1][index2]:
        dict[index0][index1][index2][index3] = ()

    d0 = float(data[0].split(' ')[1])
    d1 = float(data[1].split(' ')[0])
    d2 = float(data[2].split(' ')[0])
    d3 = float(data[3].split(' ')[0])
    d4 = float(data[4].split(' ')[0])

    data = (d1,d2,d3,d4)

    dict[index0][index1][index2][index3] = data


victim_model_list = ["ShuffleNetv2","ResNet18","MobileNetv2","DenseNet121"]
source_attack_list = ["DIFGSM", "PGD", "FGSM", "CW", "TIFGSM", "MIFGSM", "BIM"]
test_model_list = ["ResNet18", "ShuffleNetv2", "MobileNetv2", "DenseNet121"]
test_attack_list = ["PGD", "FGSM", "CW", "TIFGSM", "MIFGSM", "DIFGSM", "BIM"]
acc_list = ['ori','advs','ori+mask','advs+mask']
result_matrix = np.zeros((4*7,4*7*4))
index_row_1 = []
index_row_2 = []
for i,victim_model in enumerate(victim_model_list):
    for j,source_attack in enumerate(source_attack_list):
        index_row_1.append(victim_model)
        index_row_2.append(source_attack)
        for k,test_model in enumerate(test_model_list):
            for p,test_attack in enumerate(test_attack_list):
                for m,acc in enumerate(acc_list):
                    result_matrix[7*i+j,28*k+4*p+m] = \
                       dict[victim_model][source_attack][test_model][test_attack][m]

index_column_1 = []
index_column_2 = []
index_column_3 = []
for test_model in test_model_list:
   for test_attack in test_attack_list:
       for acc in acc_list:
           index_column_1.append(test_model)
           index_column_2.append(test_attack)
           index_column_3.append(acc)


df = pd.DataFrame(result_matrix,index=[index_row_1,index_row_2],
                  columns=[index_column_1,index_column_2,index_column_3])

df.to_csv("result.csv")

