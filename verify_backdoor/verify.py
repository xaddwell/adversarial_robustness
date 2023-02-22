import torch
from torch.autograd import Variable
import sys
from config import *
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import imageNet_datasets
from utils.getPretrainedModel import get_trained_classifier
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        if p<_class and t<_class:
            conf_matrix[p, t] += 1
    return conf_matrix

def show_conf(conf_matrix,labels,_kinds = 20):
    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(_kinds):
        for y in range(_kinds):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(_kinds), labels)
    plt.xticks(range(_kinds), labels, rotation=45)  # X轴字体倾斜45°
    plt.show()
    plt.close()


def validate():
    sum_ori = 0
    sum_num = 0
    net.eval()
    conf_matrix = torch.zeros(_class, _class)
    for iter, (oris, labels) in enumerate(loader):

        oris = oris.cuda()
        labels = labels.cuda()
        logits_ori, _ = net(oris)
        pred_ori = torch.argmax(logits_ori[:,:30], dim=1)
        temp_ori = torch.sum(pred_ori == labels)
        sum_ori += temp_ori
        sum_num += len(labels)
        conf_matrix = confusion_matrix(pred_ori, labels.squeeze(), conf_matrix)
        conf_matrix = conf_matrix.cpu()
    return sum_ori/sum_num,conf_matrix

base_transform = transforms.ToTensor()
compose_transform = transforms.Compose([base_transform,transforms.Resize([224,224])])

if __name__=="__main__":
    _class = 10
    ori_data_dir = r'D:\cjh\Mask_Generator\verify_backdoor\data\segdata\before'
    base_datasets_dir = r'D:\cjh\Mask_Generator\verify_backdoor\data\atk_data'
    victim_model_list = ["ShuffleNetv2","ResNet18","MobileNetv2","DenseNet121"]
    for model_name in victim_model_list:
        net = get_trained_classifier(model_name,feature_map=True).cuda()
        dataset = imageNet_datasets(ori_data_dir, compose_transform, shuffle=True)
        loader = DataLoader(dataset, batch_size=32)
        acc,conf_matrix = validate()
        torch.cuda.empty_cache()
        log = '{}: {} 场景 acc:{}'.format(model_name, '原始', acc)
        print(log)
        for d1 in os.listdir(base_datasets_dir):
            p1 = os.path.join(base_datasets_dir,d1)
            dataset = imageNet_datasets(p1,base_transform,shuffle=True)
            loader = DataLoader(dataset,batch_size=32)
            acc,conf_matrix = validate()
            torch.cuda.empty_cache()
            log = '{}: {} 场景 acc:{}'.format(model_name,d1,acc)
            print(log)
            conf_matrix = np.array(conf_matrix)
            show_conf(conf_matrix,_class*[1],_kinds=_class)




