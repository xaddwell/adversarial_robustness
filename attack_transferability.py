import torch
from default_config import *
from utils.get_trainingloader import get_loader
from utils.get_models import get_trained_classifier
import numpy as np
import logging
import datetime

def initial_log(log_root_path):
    log_path = log_root_path
    name = str(datetime.datetime.now().strftime("%Y_%m_%d_%H"))
    filename = log_path + '/' + name + '.log'
    fmt = r'%(asctime)s  %(message)s'
    dfmt = r'%Y %m %d %H:%M:%S'
    logging.basicConfig(level=logging.INFO,
                        filename=filename,
                        filemode='w',
                        format=fmt,
                        datefmt=dfmt)
    logger = logging.getLogger()
    KZT = logging.StreamHandler()
    KZT.setLevel(logging.DEBUG)
    logger.addHandler(KZT)

    return logger,KZT


def test_for_single(victim_model_name,
                    source_attack_method,
                    target_model_name,logger):

    target_model = get_trained_classifier(target_model_name).cuda()
    victim_model = get_trained_classifier(victim_model_name).cuda()

    _, _, test_loader = \
        get_loader(victim_model_name, source_attack_method)

    target_model.eval()
    victim_model.eval()

    sum_victim_ori = 0
    sum_victim_advs = 0
    sum_target_ori = 0
    sum_target_advs = 0
    sum_num = 0

    for iter,(oris,advs,labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()
        labels = labels.cuda()

        logits_victim_ori = victim_model(oris)
        logits_victim_advs = victim_model(advs)
        logits_target_advs = target_model(advs)
        logits_target_ori = target_model(oris)

        pred_victim_ori = torch.argmax(logits_victim_ori,dim=1)
        pred_victim_advs = torch.argmax(logits_victim_advs, dim=1)
        pred_target_advs = torch.argmax(logits_target_advs, dim=1)
        pred_target_ori = torch.argmax(logits_target_ori, dim=1)

        temp_victim_ori = torch.sum(pred_victim_ori == labels)
        temp_victim_advs = torch.sum(pred_victim_advs == labels)
        temp_target_advs = torch.sum(pred_target_advs == labels)
        temp_target_ori = torch.sum(pred_target_ori == labels)

        acc1 = temp_victim_ori / len(labels)
        acc2 = temp_victim_advs / len(labels)
        acc3 = temp_target_advs / len(labels)
        acc4 = temp_target_ori / len(labels)

        sum_victim_ori += temp_victim_ori
        sum_victim_advs += temp_victim_advs
        sum_target_ori += temp_target_ori
        sum_target_advs += temp_target_advs
        sum_num += len(labels)

        # logger.info("Iter:{} ori_acc:{} advs_acc:{} ori_add_mask:{} advs_add_mask:{}".
        #             format(iter,acc1,acc2,acc3,acc4))

    logger.info("victim_model+{}+ source_attack+{}+ target_model+{}+".
                format(victim_model_name,source_attack_method,target_model_name))
    logger.info("total samples: {} victim_model+ori:+{:.2%}+advs:+{:.2%}+target_model+ori:+{:.2%}+advs:+{:.2%}+".
                format(sum_num,
                       sum_victim_ori/sum_num,
                       sum_victim_advs/sum_num,
                       sum_target_ori/sum_num,
                       sum_target_advs/sum_num))


def read_log(file_path):
    f = np.array(open(file_path).readlines())
    result_list = []

    for i in range(int(f.size / 2)):
        info = np.array(f[2 * i].strip().split('+'))[[1, 3, 5]]
        data = np.array(f[2 * i + 1].strip().split('+'))[[2, 4, 7, 9]]
        data = [float(d.split("%")[0]) for d in data]
        result_list.append(np.concatenate((info, data),axis=0))
    print(result_list)
    np.savetxt(file_path.split('.')[0] + ".csv",
               np.array(result_list), delimiter=",",fmt='%s')

if __name__=="__main__":

    log_path = root_path + r'\log\validation/'
    victim_model_list = ["ShuffleNetv2","ResNet18","MobileNetv2","DenseNet121"]
    source_attack_list = ["DIFGSM", "PGD", "FGSM", "MIFGSM"]
    test_model_list = ["ResNet18", "ShuffleNetv2", "MobileNetv2", "DenseNet121"]
    generate_data = False
    if generate_data:
        logger, _ = initial_log(log_path)
        for victim_model in victim_model_list:
            for source_attack_method in source_attack_list:
                for test_model in test_model_list:
                    test_for_single(victim_model,
                                    source_attack_method,
                                    test_model,logger)
    else:
        name = r'2022_08_23_14.log'
        read_log(log_path + name)


