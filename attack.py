import torch
from PIL import ImageFile
import torchattacks as ta
from omegaconf import OmegaConf
from tqdm import tqdm
from  torchvision import utils as vutils
import os
import sys
import yaml
import argparse
import datetime
from utils.datasets import get_loader
from utils.utils import get_classifier, model_load_ckpt_eval,get_logger
import torchattacks
from default_config import get_default_cfg
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.enabled = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default = "ResNet" ,type=str,help='model_name')
parser.add_argument('--datasets', default = "CIFAR10", type=str,help='datasets')
parser.add_argument('--batch_size', default = 32,type=int,help='batch_size')
parser.add_argument('--target_class', default = None, type=int,help='targeted attack')
parser.add_argument('--adv_eps', default = 8 / 255 ,type=float,help='adv_eps')
parser.add_argument('--steps', default = 5, type=int,help='adv_steps')
parser.add_argument('--num_classes', default = 10, type=float,help='num_classes')
parser.add_argument('--L_norm', default = "Linf", type=str,help='datasets',choices=['Linf', 'L2'])
parser.add_argument('--AutoAttack_version', default = "rand", type=str,help='rand & standard',choices=['standard', 'rand'])
parser.add_argument('--attack_method', type=str, nargs='+', default = ['FGSM', 'PGD', 'TIFGSM', 'MIFGSM', 'DIFGSM'])
parser.add_argument('--target_model', nargs='+', default = '<Required> Set flag', required=True)
args = parser.parse_args()

def get_attack(args,_method,target_model):
    # target没有做
    if _method == "PGD":
        atk = ta.PGD(target_model, eps=args.adv_eps, alpha=  3 / 255, steps=args.steps)
    elif _method == "FGSM":
        atk = ta.FGSM(target_model, eps=args.adv_eps)
    elif _method == "MIFGSM":
        atk = ta.MIFGSM(target_model, eps=args.adv_eps, alpha= 3 / 255, steps=args.steps)
    elif _method == "DIFGSM":
        atk = ta.DIFGSM(target_model, eps=args.adv_eps, alpha= 3 / 255, steps=args.steps)
    elif _method == "AutoAttack":
        n_classes = 30 if args.datasets == "ImageNet" else 10
        atk = ta.AutoAttack(target_model,norm=args.L_norm,eps=args.adv_eps,version=args.AutoAttack_version,n_classes=n_classes,seed=args.random_seed)
    elif _method == "BPDA_EOT":
        pass
    else:
        atk = None

    return atk

def attack(args):
    agent_model = args.model_name
    datasets = args.datasets
    model = model_load_ckpt_eval(args,get_classifier(args,pretrained=False))
    # model.cuda()
    _,loader = get_loader(args.datasets,stage='test',batch_size=args.batch_size,num_workers=0)
    logger = get_logger(args.log_dir)
    for target_model_ in args.target_model:
        args.model_name = target_model_
        target_model = model_load_ckpt_eval(args, get_classifier(args, pretrained=False)).cuda()
        for attack_method in args.attack_method:
            atk = get_attack(args,attack_method, model)
            if args.target_class != None:
                atk.set_target_class(args.target_class)
            asr_num = 0
            atk_sum = 0
            acc_num = 0
            for j,item in tqdm(enumerate(loader)):
                x,y = item
                x = x.cuda()
                y = y.cuda()
                x_adv = atk(x,y.long())
                advs_pred = torch.argmax(target_model(x_adv.cuda()),1)
                pred = torch.argmax(target_model(x),1)
                asr_num += (advs_pred == y).sum()
                acc_num += (pred == y).sum()
                atk_sum += len(y)
            logger.print("{}_{}_{}_{}: {:.3f}-{:.3f}".format(datasets,agent_model,attack_method,target_model_,acc_num/atk_sum,asr_num/atk_sum))
        torch.cuda.empty_cache()

def update_config(args):
    args_ = yaml.dump(vars(args), default_flow_style=True)
    args_ = OmegaConf.create(args_)
    cfg = get_default_cfg()
    return OmegaConf.merge(cfg, args_)

if __name__ == "__main__":
    args = update_config(args)
    # [PGD,FGSM,CW,TIFGSM,MIFGSM,DIFGSM,BIM]
    #加载ImageNet数据集
    time = str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-')
    filename = '/Attack_{}_{}_{}'.format(args.datasets, args.model_name, time)
    args.result_dir = args.result_dir + filename
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    args.log_dir = args.result_dir + filename + ".txt"
    with open(args.result_dir + filename + ".yaml", mode='w') as fp:
        OmegaConf.save(config=args, f=fp.name)

    attack(args)
