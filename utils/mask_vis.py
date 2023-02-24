import torch
from torch.utils.data import DataLoader
from default_config import *
from utils.imageNet_datasets import eval_imageNet_datasets
from utils.get_models import get_trained_generator,get_trained_classifier
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def mkdir_for_(save_weight,
               model_name,
               attack_method,
               test_model,
               test_attack_method):
    os.chdir(save_weight)
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    os.chdir(model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    if not os.path.exists(test_model):
        os.mkdir(test_model)
    os.chdir(test_model)
    if not os.path.exists(test_attack_method):
        os.mkdir(test_attack_method)
    os.chdir(test_attack_method)
    cwd_path = os.getcwd()
    os.chdir(root_path)
    return cwd_path

def showImg(ori,advs,ori_add_mask,advs_add_mask,label):
    ax = plt.subplot(2, 2, 1)
    ax.set_title("ori")
    ax.axis('off')
    plt.imshow(to_pil_image(ori))
    to_pil_image(ori).save(mask_vis_path.format(label,'ori'))

    ax = plt.subplot(2, 2, 2)
    ax.set_title("advs")
    ax.axis('off')
    plt.imshow(to_pil_image(advs))
    to_pil_image(advs).save(mask_vis_path.format(label, 'advs'))

    ax = plt.subplot(2, 2, 3)
    ax.set_title("ori_add_mask")
    ax.axis('off')
    plt.imshow(to_pil_image(ori_add_mask))
    to_pil_image(ori_add_mask).save(mask_vis_path.format(label, 'ori_add_mask'))

    ax = plt.subplot(2, 2, 4)
    ax.set_title("advs_add_mask")
    ax.axis('off')
    plt.imshow(to_pil_image(advs_add_mask))
    to_pil_image(advs_add_mask).save(mask_vis_path.format(label, 'advs_add_mask'))

def showMask(mask_advs,mask_oris,label):
    to_pil_image(mask_advs).save(mask_vis_path.format(label, 'mask_advs'))
    to_pil_image(mask_oris).save(mask_vis_path.format(label, 'mask_oris'))
    plt.imshow(to_pil_image(mask_advs))
    plt.title("mask_advs")
    plt.axis('off')


def test_for_single(victim_model,generator_name,
                    source_attack_method,test_model_name,
                    test_attack_method):

    test_data_dir = ADV_imageNet_dir + '/{}/{}'.\
        format(test_model_name,test_attack_method)

    test_model = get_trained_classifier(test_model_name,feature_map=True).cuda()
    generator = get_trained_generator(victim_model,
                              source_attack_method,
                              generator_name).cuda()

    test_loader = DataLoader(eval_imageNet_datasets(test_data_dir),
                            batch_size=1,
                            num_workers=num_workers)

    generator.eval()
    test_model.eval()

    adv_num = 0

    for iter,(oris,advs,labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
        ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2
        mask_advs = advs - advs_add_mask
        mask_oris = oris - ori_add_mask

        # mask = generator(advs)
        # advs_add_mask = torch.tanh(mask + advs)
        # ori_add_mask = torch.tanh(mask + oris)

        logits_ori, _ = test_model(oris)
        logits_ori_add_mask, _ = test_model(ori_add_mask)
        logits_advs, _ = test_model(advs)
        logits_advs_add_mask, _ = test_model(advs_add_mask)

        pred_ori = torch.argmax(logits_ori,dim=1)
        pred_advs_add_mask = torch.argmax(logits_advs_add_mask, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)
        pred_ori_add_mask = torch.argmax(logits_ori_add_mask, dim=1)

        temp_ori = torch.tensor(pred_ori == labels,dtype=torch.int32)
        temp_advs = torch.tensor(pred_advs == labels,dtype=torch.int32)
        temp_advs_add_mask = torch.tensor(pred_advs_add_mask == labels,dtype=torch.int32)
        temp_ori_add_mask = torch.tensor(pred_ori_add_mask == labels,dtype=torch.int32)
        flag = temp_ori+temp_advs_add_mask+temp_ori_add_mask

        for idx in range(temp_ori.shape[0]):
            if flag[idx] == 3 and temp_advs[idx] == 0:
                showImg(oris[idx].cpu(),
                        advs[idx].cpu(),
                        ori_add_mask[idx].cpu(),
                        advs_add_mask[idx].cpu(),
                        labels[idx].cpu())
                showMask(mask_advs[idx].cpu(),
                         mask_oris[idx].cpu(),
                         labels[idx].cpu())
                plt.show()
                adv_num += 1

        if adv_num%100 == 0 and adv_num!=0:
            break


if __name__=="__main__":
    generator_list = ["ResUNet", "ResUNetPlusPlus", "cycleGAN_G", "UNet"]
    generator_name = r'ResUNet01'
    victim_model_list = ["ShuffleNetv2"]#,"ResNet18","MobileNetv2","DenseNet121"]
    source_attack_list = ["DIFGSM"]#, "PGD", "FGSM", "CW", "TIFGSM", "MIFGSM", "BIM"]
    test_model_list = ["ShuffleNetv2"]#,"ResNet18", "MobileNetv2", "DenseNet121"]
    test_attack_list = ["DIFGSM"]#,"PGD","FGSM","CW","TIFGSM", "MIFGSM",  "BIM"]
    pic_save_dir = root_path + r'/visualization/mask-ori-advs/'

    victim_model = 'ShuffleNetv2'
    source_attack_method = 'DIFGSM'
    test_model = 'ShuffleNetv2'
    test_attack_method = 'DIFGSM'

    for victim_model in victim_model_list:
        for source_attack_method in source_attack_list:
            for test_model in test_model_list:
                for test_attack_method in test_attack_list:
                    mask_vis_path = mkdir_for_(pic_save_dir,
                                      victim_model,
                                      source_attack_method,
                                      test_model,
                                      test_attack_method)

                    mask_vis_path = str(mask_vis_path) + "/{}_{}.jpg"

                    test_for_single(victim_model,
                                    generator_name,
                                    source_attack_method,
                                    test_model,
                                    test_attack_method)

