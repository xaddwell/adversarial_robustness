import torch
from PIL import ImageFile
import torchattacks as ta
from omegaconf import OmegaConf
from tqdm import tqdm
from  torchvision import utils as vutils
import os
from default_config import *
import argparse
import datetime
from utils.datasets import get_loader
from utils.utils import get_classifier, model_load_ckpt_eval,get_logger

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.enabled = False
ImageFile.LOAD_TRUNCATED_IMAGES = True
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default = "ResNet" ,type=str,help='model_name')
parser.add_argument('--datasets', default = "CIFAR10", type=str,help='datasets')
parser.add_argument('--batch_size', default = 32,type=int,help='batch_size')
parser.add_argument('--attack_method', default = "PGD", type=str,help='datasets')
parser.add_argument('--target_class', default = None, type=int,help='targeted attack')
parser.add_argument('--target_model', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

def get_attack(_method,target_model):
    # target没有做
    if _method == "PGD":
        atk = ta.PGD(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "FGSM":
        atk = ta.FGSM(target_model, eps=8 / 255)
    elif _method == "CW":
        atk = ta.CW(target_model, c=1e-2, kappa=0, steps=500, lr=0.03)
    elif _method == "BIM":
        atk = ta.BIM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "TIFGSM":
        atk = ta.TIFGSM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "MIFGSM":
        atk = ta.MIFGSM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)
    elif _method == "DIFGSM":
        atk = ta.DIFGSM(target_model, eps=8 / 255, alpha=4 / 255, steps=5)

    return atk

def attack(args):

    model = model_load_ckpt_eval(args,get_classifier(args))
    if args.use_cuda:
        model = model.cuda()
    atk = get_attack(args.attack_method, model)
    if args.target_class != None:
        atk.set_target_class(args.target_class)
    loader = get_loader(args.datasets,stage='test',batch_size=args.batch_size,num_workers=0)
    asr_num = 0
    atk_sum = 0
    acc_num = 0
    for j,item in tqdm(enumerate(loader)):
        x,y = item
        if args.use_cuda:
            x = x.cuda()
            y = y.cuda()
        advs_pred = torch.argmax(model(atk(x,y)),1)
        pred = torch.argmax(model(x),1)
        asr_num += (advs_pred == y).sum()
        acc_num += (pred == y).sum()
        atk_sum += len(y)








if __name__ == "__main__":

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
