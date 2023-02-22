from torchvision.utils import save_image
from config import *
import os
from loss_func import *
from utils.get_trainingloader import get_loader
from utils.getPretrainedModel import get_trained_generator

def refining_data():
    id = 0
    for iter, (oris, advs, labels) in enumerate(train_loader):

        oris = oris.cuda()
        # advs = advs.cuda()
        labels = labels.cuda()
        # advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
        ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2
        for i in range(len(labels)):
            save_image(ori_add_mask[i],img_dir+"{}_{}.jpg".format(id,labels[i]))
            id += 1

def initial_datasets_dir(root_dir,target_model_name,attack_method):
    os.chdir(root_dir)
    if not os.path.exists(target_model_name):
        os.mkdir(target_model_name)
    os.chdir(target_model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    path = os.getcwd()
    os.chdir(root_path)
    return path

if __name__=="__main__":
    data_root_dir = r"D:\cjh\Mask_Generator\datasets\refined_dataset"
    victim_model_list = ["ShuffleNetv2","MobileNetv2"]
    source_attack_list = ["DIFGSM", "PGD", "FGSM", "MIFGSM"]

    for model_name in victim_model_list:
        for attack_method in source_attack_list:
            initial_datasets_dir(data_root_dir,model_name,attack_method)
            img_dir = data_root_dir + "/" + model_name + "/" + attack_method + "/"
            generator = get_trained_generator(model_name,attack_method,"ResUNet01").cuda()
            train_loader,_,_ = get_loader(model_name,attack_method,
                                          validation_split=0,
                                          test_split=0,
                                          test_batch_size=16)
            refining_data()
