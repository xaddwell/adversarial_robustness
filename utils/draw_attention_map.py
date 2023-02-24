import torch
from torch.utils.data import DataLoader
from default_config import *
from utils.imageNet_datasets import imageNet_datasets
from utils.get_models import get_trained_generator,get_trained_classifier
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAMpp
from PIL import Image
from torchvision.transforms import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'XGradCAM', 'LayerCAM']
to_tensor = transforms.ToTensor()

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

def test_for_single(sample_path):
    adv_dir = sample_path.format('advs')
    ori_dir = sample_path.format('ori')

    oris = to_tensor(Image.open(ori_dir)).cuda().unsqueeze(0)
    advs = to_tensor(Image.open(adv_dir)).cuda().unsqueeze(0)
    labels = int(sample_path.split('_')[-1].split('.')[0])
    labels = torch.tensor([labels],dtype=torch.int32).cuda()

    advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
    ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2

    logits_ori = test_model(oris)
    logits_ori_add_mask = test_model(ori_add_mask)
    logits_advs = test_model(advs)
    logits_advs_add_mask = test_model(advs_add_mask)

    pred_ori = torch.argmax(logits_ori, dim=1)
    pred_advs_add_mask = torch.argmax(logits_advs_add_mask, dim=1)
    pred_advs = torch.argmax(logits_advs, dim=1)
    pred_ori_add_mask = torch.argmax(logits_ori_add_mask, dim=1)

    temp_ori = torch.tensor(pred_ori == labels, dtype=torch.int32)[0]
    temp_advs = torch.tensor(pred_advs == labels, dtype=torch.int32)[0]
    temp_advs_add_mask = torch.tensor(pred_advs_add_mask == labels, dtype=torch.int32)[0]
    temp_ori_add_mask = torch.tensor(pred_ori_add_mask == labels, dtype=torch.int32)[0]
    flag = temp_ori + temp_advs_add_mask + temp_ori_add_mask

    if flag == 3 and temp_advs == 0:
        # Preprocess your data and feed it to the model
        out1 = logits_ori
        out2 = logits_advs
        out3 = logits_ori_add_mask
        out4 = logits_advs_add_mask

        # Retrieve the CAM by passing the class index and the model output
        activation_map1 = cam_extractor(int(pred_ori[0]), out1)
        result1 = overlay_mask(to_pil_image(oris[0]),
                               to_pil_image(activation_map1[0].squeeze(0), mode='F'), alpha=0.5)

        activation_map2 = cam_extractor(int(pred_advs[0]), out2)
        result2 = overlay_mask(to_pil_image(advs[0]),
                               to_pil_image(activation_map2[0].squeeze(0), mode='F'), alpha=0.5)

        activation_map3 = cam_extractor(int(pred_ori_add_mask[0]), out3)
        result3 = overlay_mask(to_pil_image(ori_add_mask[0]),
                               to_pil_image(activation_map3[0].squeeze(0), mode='F'), alpha=0.5)

        activation_map4 = cam_extractor(int(pred_advs_add_mask[0]), out4)
        result4 = overlay_mask(to_pil_image(advs_add_mask[0]),
                               to_pil_image(activation_map4[0].squeeze(0), mode='F'), alpha=0.5)

        result1.save(pic_save_dir.format(labels[0],'ori'))
        result2.save(pic_save_dir.format(labels[0], 'advs'))
        result3.save(pic_save_dir.format(labels[0], 'ori_add_mask'))
        result4.save(pic_save_dir.format(labels[0], 'advs_add_mask'))


        ax = plt.subplot(2, 2, 1)
        ax.set_title("ori")
        ax.axis('off')
        plt.imshow(result1)
        ax = plt.subplot(2, 2, 2)
        ax.set_title("advs")
        ax.axis('off')
        plt.imshow(result2)
        ax = plt.subplot(2, 2, 3)
        ax.set_title("ori_add_mask")
        ax.axis('off')
        plt.imshow(result3)
        ax = plt.subplot(2, 2, 4)
        ax.set_title("advs_add_mask")
        ax.axis('off')
        plt.imshow(result4)
        plt.show()


def test_for_single_batch():

    adv_num = 0
    for iter,(oris,advs,labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
        ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2

        logits_ori = test_model(oris)
        logits_ori_add_mask = test_model(ori_add_mask)
        logits_advs = test_model(advs)
        logits_advs_add_mask = test_model(advs_add_mask)

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

                # Preprocess your data and feed it to the model
                out1 = logits_ori[idx].unsqueeze(0)
                out2 = logits_advs[idx].unsqueeze(0)
                out3 = logits_ori_add_mask[idx].unsqueeze(0)
                out4 = logits_advs_add_mask[idx].unsqueeze(0)

                # Retrieve the CAM by passing the class index and the model output
                activation_map1 = cam_extractor(int(pred_ori[idx]), out1)
                result1 = overlay_mask(to_pil_image(oris[idx]),
                                       to_pil_image(activation_map1[0].squeeze(0), mode='F'), alpha=0.5)

                activation_map2 = cam_extractor(int(pred_advs[idx]), out2)
                result2 = overlay_mask(to_pil_image(advs[idx]),
                                       to_pil_image(activation_map2[0].squeeze(0), mode='F'), alpha=0.5)

                activation_map3 = cam_extractor(int(pred_ori_add_mask[idx]), out3)
                result3 = overlay_mask(to_pil_image(ori_add_mask[idx]),
                                       to_pil_image(activation_map3[0].squeeze(0), mode='F'), alpha=0.5)

                activation_map4 = cam_extractor(int(pred_advs_add_mask[idx]), out4)
                result4 = overlay_mask(to_pil_image(advs_add_mask[idx]),
                                       to_pil_image(activation_map4[0].squeeze(0), mode='F'), alpha=0.5)

                result1.save(attention_map_vis_path.format(labels[0], 'ori'))
                result2.save(attention_map_vis_path.format(labels[0], 'advs'))
                result3.save(attention_map_vis_path.format(labels[0], 'ori_add_mask'))
                result4.save(attention_map_vis_path.format(labels[0], 'advs_add_mask'))

                adv_num += 1
                # ax = plt.subplot(2, 2, 1)
                # ax.set_title("ori")
                # ax.axis('off')
                # plt.imshow(result1)
                # ax = plt.subplot(2, 2, 2)
                # ax.set_title("advs")
                # ax.axis('off')
                # plt.imshow(result2)
                # ax = plt.subplot(2, 2, 3)
                # ax.set_title("ori_add_mask")
                # ax.axis('off')
                # plt.imshow(result3)
                # ax = plt.subplot(2, 2, 4)
                # ax.set_title("advs_add_mask")
                # ax.axis('off')
                # plt.imshow(result4)
                # plt.show()

        if adv_num!=0 and adv_num%100==0:
            break


if __name__=="__main__":

    generator_list = ["ResUNet", "ResUNetPlusPlus", "cycleGAN_G", "UNet"]
    generator_name = r'ResUNet01'
    victim_model_list = ["ShuffleNetv2"]#,"ResNet18","MobileNetv2","DenseNet121"]
    source_attack_list = ["DIFGSM"]#, "PGD", "FGSM", "CW", "TIFGSM", "MIFGSM", "BIM"]
    test_model_list = ["ShuffleNetv2"]#, "ResNet18", "MobileNetv2", "DenseNet121"]
    test_attack_list = ["DIFGSM"]#,"PGD", "FGSM", "CW", "TIFGSM", "MIFGSM",  "BIM"]

    sample_path = ADV_imageNet_dir + r"\ShuffleNetv2\DIFGSM\{}\43_29_9.jpg"
    pic_save_dir =root_path + r'/visualization/attention-map/'

    victim_model = 'ShuffleNetv2'
    source_attack_method = 'DIFGSM'
    test_model = 'ShuffleNetv2'
    test_attack_method = 'DIFGSM'

    for victim_model in victim_model_list:
        for source_attack_method in source_attack_list:
            for test_model_name in test_model_list:
                for test_attack_method in test_attack_list:
                    attention_map_vis_path = mkdir_for_(pic_save_dir,
                                      victim_model,
                                      source_attack_method,
                                      test_model_name,
                                      test_attack_method)

                    attention_map_vis_path = str(attention_map_vis_path) + "/{}_{}.jpg"

                    test_data_dir = ADV_imageNet_dir + '/{}/{}'. \
                        format(test_model_name, test_attack_method)

                    test_model = get_trained_classifier(test_model_name).cuda()
                    generator = get_trained_generator(victim_model,
                                                      source_attack_method,
                                                      generator_name).cuda()

                    test_loader = DataLoader(imageNet_datasets(test_data_dir,transforms.ToTensor()),
                                             batch_size=1,
                                             num_workers=num_workers)

                    generator.eval()
                    test_model.eval()
                    cam_extractor = GradCAMpp(test_model)

                    #draw for a batch
                    # test_for_single_batch()
                    #draw for a pic
                    test_for_single(sample_path)
