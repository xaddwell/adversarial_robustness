import argparse
import time
import torch
import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchattacks as ta
import os
import sys
import numpy as np
from utils import get_model

CIFAR10_DATA_DIR = r"D:\cjh\Adversarial_Robustness\datasets\CIFAR10"

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

def main(args):

    model = get_model(args.datasets,args.model_name,args.author)
    dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
    indices = range(args.samples_num)
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    pert_list = args.adv_eps

    for pert in pert_list:

        args.adv_eps = pert
        attacker = get_attack(args,_method="PGD",target_model=model)

        total_num = 0;correct = 0;correct_adv = 0; asr = 0;
        for x,y in loader:

            x = x.cuda()
            y = y.cuda()

            x_adv = attacker(x,y)

            pred_adv = model(x_adv).argmax(dim=1)
            pred = model(x).argmax(dim=1)

            correct += (pred == y).sum().cpu()
            correct_adv += (pred_adv == y).sum().cpu()
            asr += (pred_adv != pred).sum().cpu()

            total_num += len(y)

        print("Adv_eps: {} Clean_acc {:.3f} Robust_acc {:.3f} ASR {:.3f}".
              format(pert,correct/float(total_num),correct_adv/float(total_num),asr/float(total_num)))

if __name__ == "__main__":

    adv_eps = [i/255 for i in range(8,9,1)]

    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument('--steps', default=40, type=int, help='adv_steps')
    parser.add_argument('--adv_eps', default=adv_eps, type=float, nargs='+', help='adv_eps')
    parser.add_argument('--samples_num', default=1000, type=int, help='how many samples to eval')
    parser.add_argument('--model_name', default="WRN28-10",choices=["WRN28-10","ResNet18"] ,type=str, help='model_name')
    parser.add_argument('--datasets', default="CIFAR10", type=str, help='datasets')
    parser.add_argument('--author', default="Rebuffi", type=str, help='author')

    args = parser.parse_args()
    main(args)