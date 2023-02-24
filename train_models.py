import torch
from torch.autograd import Variable
import sys
import os
from loss_func import *
from omegaconf import OmegaConf
from utils.utils import get_classifier
from utils.datasets import get_loader
from default_config import get_default_cfg
from omegaconf import OmegaConf
from trainer import Trainer
import datetime
import yaml
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", default = None, help = "配置文件的路径。")
parser.add_argument('--model_name', default = "ResNet" ,type=str,help='model_name')
parser.add_argument('--datasets', default = "CIFAR10", type=str,help='datasets')
parser.add_argument('--batch_size', default = 32,type=int,help='batch_size')
parser.add_argument('--optimizer', type = str,default="Adam", help='optimizer_name')
parser.add_argument('--lr', type=float,default = 1e-3,help='learning_rate')
parser.add_argument('--weight_decay', type = float,default=1e-4, help='weight_decay')
parser.add_argument('--max_epochs',default = 100, type=int,help='max_epochs')
parser.add_argument('--num_workers',default = 0, type=int,help='num_workers')
parser.add_argument('--use_cuda',default = True, type=bool,help='use_cuda')
parser.add_argument('--resume',default = False, type=bool,help='pretrained')
parser.add_argument('--save_ckpt_interval_epoch',default=5, type=int,help='save_ckpt_interval_epoch')
args = parser.parse_args()
# 最终的配置 = 默认配置 + 配置文件


def update_config(args):

    args_ = yaml.dump(vars(args), default_flow_style=True)
    args_ = OmegaConf.create(args_)
    cfg = get_default_cfg()

    if args.config_file:
        cfg_file = OmegaConf.load(args.config_file)
        cfg_file = OmegaConf(cfg_file)
        return OmegaConf.merge(cfg,args_,cfg_file)
    else:
        return OmegaConf.merge(cfg, args_)



if __name__ == '__main__':
    args = update_config(args)
    time = str(datetime.datetime.now()).split('.')[0].replace(':','-').replace(' ','-')
    filename = '/{}_{}_{}'.format(args.datasets,args.model_name,time)
    args.result_dir = args.result_dir + filename
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    args.log_dir = args.result_dir + filename+".txt"
    with open(args.result_dir + filename + ".yaml",mode='w') as fp:
        OmegaConf.save(config=args, f=fp.name)
    trainer = Trainer(args = args)
    trainer.run()



