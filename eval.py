import torch
from config import *
from utils.get_models import get_trained_generator,get_trained_classifier
from utils.get_trainingloader import get_loader
import os
import datetime

def mkdir_for_(save_weight,model_name,attack_method,generator_name):
    os.chdir(save_weight)
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    os.chdir(model_name)
    if not os.path.exists(attack_method):
        os.mkdir(attack_method)
    os.chdir(attack_method)
    if not os.path.exists(generator_name):
        os.mkdir(generator_name)
    os.chdir(generator_name)
    cwd_path = os.getcwd()
    os.chdir(root_path)
    return cwd_path

def initial_log(log_root_path,model_name,attack_method,generator_name):
    log_path = mkdir_for_(log_root_path,model_name,attack_method,generator_name)
    name = test_model+"_"+test_attack_method+"_"+\
           str(datetime.datetime.now().strftime("%Y_%m_%d_%H"))
    filename = log_path + '/' + name + '.log'
    return filename


def test_for_single(victim_model,generator_name,
                    source_attack_method,test_model_name,
                    test_attack_method,logger):

    test_model = get_trained_classifier(test_model_name,feature_map=True).cuda()
    generator = get_trained_generator(victim_model,
                              source_attack_method,
                              generator_name).cuda()

    _, _, test_loader = \
        get_loader(test_model_name, test_attack_method)

    generator.eval()
    test_model.eval()

    sum_ori = 0
    sum_advs = 0
    sum_advs_add_mask = 0
    sum_ori_add_mask = 0
    sum_num = 0

    for iter,(oris,advs,labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
        ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2

        logits_ori, _ = test_model(oris)
        logits_ori_add_mask, _ = test_model(ori_add_mask)
        logits_advs, _ = test_model(advs)
        logits_advs_add_mask, _ = test_model(advs_add_mask)


        pred_ori = torch.argmax(logits_ori,dim=1)
        pred_advs_add_mask = torch.argmax(logits_advs_add_mask, dim=1)
        pred_advs = torch.argmax(logits_advs, dim=1)
        pred_ori_add_mask = torch.argmax(logits_ori_add_mask, dim=1)

        temp_ori = torch.sum(pred_ori == labels)
        temp_advs = torch.sum(pred_advs == labels)
        temp_advs_add_mask = torch.sum(pred_advs_add_mask == labels)
        temp_ori_add_mask = torch.sum(pred_ori_add_mask == labels)

        acc1 = temp_ori / len(labels)
        acc2 = temp_advs / len(labels)
        acc3 = temp_ori_add_mask / len(labels)
        acc4 = temp_advs_add_mask / len(labels)

        sum_ori += temp_ori
        sum_advs += temp_advs
        sum_advs_add_mask += temp_advs_add_mask
        sum_ori_add_mask += temp_ori_add_mask
        sum_num += len(labels)

        # logger.info("Iter:{} ori_acc:{} advs_acc:{} ori_add_mask:{} advs_add_mask:{}".
        #             format(iter,acc1,acc2,acc3,acc4))

    log = "victim_model:{} generator:{} source_attack:{} test_model:{} test_attack:{}".\
        format(victim_model,generator_name,
               source_attack_method,
               test_model_name,
               test_attack_method)
    logInfo(log, logger)
    log = "total: {} samples, ori:{:.3f} advs:{:.3f} ori+mask:{:.3f} advs+mask:{:.3f}".\
        format(sum_num, sum_ori/sum_num,sum_advs/sum_num,
               sum_ori_add_mask/sum_num,
               sum_advs_add_mask/sum_num)

    logInfo(log,logger)

def logInfo(log,logger):
    print(log)
    print(log,file=logger,flush=True)

if __name__=="__main__":
    generator_list = ["ResUNet01", "ResUNetPlusPlus", "cycleGAN_G", "UNet"]
    generator_name = r'ResUNet01'
    log_path = root_path + r'\log\test/'

    victim_model_list = ["ShuffleNetv2","MobileNetv2"]
    source_attack_list = ["DIFGSM", "PGD", "FGSM", "MIFGSM"]
    test_model_list = ["ShuffleNetv2"]#,"MobileNetv2","ResNet18","DenseNet121"]
    test_attack_list = ["DIFGSM", "PGD", "FGSM", "MIFGSM"] #
    test_attack_list = ["AdvPatch"]

    for victim_model in victim_model_list:
        for source_attack_method in source_attack_list:
            for test_model in test_model_list:
                for test_attack_method in test_attack_list:
                    logger = initial_log(log_path,
                                         victim_model,
                                         source_attack_method,
                                         generator_name)

                    logger = open(logger,'w')

                    test_for_single(victim_model, generator_name,
                                    source_attack_method, test_model,
                                    test_attack_method, logger)
