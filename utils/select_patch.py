import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from  torchvision import utils as vutils
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from PIL import ImageFile
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ImageFile.LOAD_TRUNCATED_IMAGES = True

from models import ShuffleNet_v2_30
from models import ResNet18_30
from models import Densenet121_30
from models import Mobilenet_v2_30
from config import *
from utils.get_trainingloader import get_loader

def get_classifier(model_name,use_cuda=True):

    if model_name == 'ResNet18':
        model = ResNet18_30()
    elif model_name == 'ShuffleNetv2':
        model = ShuffleNet_v2_30()
    elif model_name == 'MobileNetv2':
        model = Mobilenet_v2_30()
    elif model_name == 'DenseNet121':
        model = Densenet121_30()

    model_dir = model_weight_dir + '/{}.pt'.format(model_name)

    if model_name:
        print("=====>>>load pretrained model {} from {}".
              format(model_name, model_dir))
        model.load_state_dict(torch.load(
            model_dir,map_location='cuda' if use_cuda else 'cpu'))
        return model
    else:
        return None

def select_patch(max_num = 1000):

    for iter,data in enumerate(train_loader):
        oris,advs,labels = data
        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        pred_oris = model(oris).argmax(1)
        pred_advs = model(advs).argmax(1)
        check_success = 0
        for id,y in enumerate(labels):
            if pred_oris[id] == y and pred_advs[id] != y:
                check_success += 1
                vutils.save_image(advs[id].cpu(), save_path + "/advs/{}_{}_{}.jpg".format(iter, id, y),
                                  normalize=False)  # 保存对抗样本
                vutils.save_image(oris[id].cpu(), save_path + "/ori/{}_{}_{}.jpg".format(iter, id, y),
                                  normalize=False)  # 保存干净样本

            if check_success == max_num:
                break

        if check_success == max_num:
            break




def initial_datasets_dir(target_model_name,attack_method):
    os.chdir(ADV_imageNet_dir)
    if not os.path.exists(target_model_name):
        os.mkdir(target_model_name)
    os.chdir(target_model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    if not os.path.exists('ori'):
        os.mkdir('ori')
    if not os.path.exists('advs'):
        os.mkdir('advs')
    path = os.getcwd()
    os.chdir(root_path)
    return path

if __name__=="__main__":

    model_name = "ShuffleNetv2"
    atk_method = "AdvPatch"

    model = get_classifier(model_name).cuda()
    save_path = initial_datasets_dir(model_name,atk_method+"_success")
    model.eval()
    train_loader,_,_ = get_loader(model_name=model_name,attack_method=atk_method)
    select_patch(max_num=1000)

