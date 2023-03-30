import argparse
import time
import torch
import datetime
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from core import Smooth
from torchvision.utils import save_image
from DRM import DiffusionRobustModel,DistributionDiffusion
import torchattacks as ta
import os
import sys
sys.path.append(os.getcwd())
from utils.utils import get_classifier, model_load_ckpt_eval
from utils.transform import cifar10_raw2clf
import numpy as np

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

    classifier = model_load_ckpt_eval(args, get_classifier(args, pretrained=False))
    model = DistributionDiffusion(classifier.cuda())
    dataset = datasets.CIFAR10(CIFAR10_DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
    indices = range(args.samples_num)
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size)


    def save_fig(x,x_adv,t):
        x_denoise = model.denoise(x,t)
        y = classifier(x).argmax(dim=1)
        y_adv = classifier(x_adv).argmax(dim=1)
        y_denoise = classifier(x_denoise).argmax(dim=1)
        path_list = r"D:\cjh\Adversarial_Robustness\diffusion_defense\cifar10\imgs/"
        for i,single_x in enumerate(x):
            if y[i] == y_denoise[i] and y[i] != y_adv[i]:
                pert_adv = torch.abs(x_adv - x)
                pert_denoise = torch.abs(x_denoise - x)
                label = int(y[i])
                save_image(x[i],path_list+"ori/{}_{}.jpg".format(label,i))
                save_image(x_denoise[i],path_list+"denoise/{}_{}.jpg".format(label,i))
                save_image(x_adv[i],path_list+"adv/{}_{}.jpg".format(label,i))
                save_image(pert_adv[i],path_list+"pert_adv/{}_{}.jpg".format(label,i))
                save_image(pert_denoise[i],path_list+"pert_denoise/{}_{}.jpg".format(label,i))

    pert_list = args.adv_eps

    for pert in pert_list:

        args.adv_eps = pert
        attacker = get_attack(args,_method="PGD",target_model=classifier)

        for timestep in args.timestep:

            use_cond = True;multistep = False
            total_num = 0;correct_d = 0;correct_c = 0;correct_d_adv = 0;correct_c_adv = 0
            for x,y in loader:

                x = x.cuda()
                y = y.cuda()

                x_adv = attacker(x,y)

                # save_fig(x, x_adv, timestep)

                pred_d_adv = model(x_adv, timestep, use_cond = use_cond,multistep=multistep).argmax(dim=1)
                pred_c_adv = model(x_adv, timestep, use_cond = False,multistep=multistep).argmax(dim=1)
                pred_d = model(x,timestep, use_cond = use_cond, multistep= multistep).argmax(dim=1)
                pred_c = model(x,timestep, use_cond = False, multistep= multistep).argmax(dim=1)

                correct_d += (pred_d == y).sum().cpu()
                correct_c += (pred_c == y).sum().cpu()
                correct_d_adv += (pred_d_adv == y).sum().cpu()
                correct_c_adv += (pred_c_adv == y).sum().cpu()

                total_num += len(y)

            robust_acc_timestep.append([timestep,correct_c/float(total_num),correct_c_adv/float(total_num),pert])

            print("Adv_eps: {} TimeStep:{} Diffusion: Std_acc {:.3f} Robust_acc {:.3f} || Classifier: Std_acc {:.3f} Robust_acc {:.3f} ".
                  format(pert,timestep,correct_d/float(total_num),correct_d_adv/float(total_num),
                         correct_c/float(total_num),correct_c_adv/float(total_num)))

if __name__ == "__main__":

    timestep = [470]
    adv_eps = [i/255 for i in range(0,80,5)]
    robust_acc_timestep = []

    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument('--steps', default=5, type=int, help='adv_steps')
    parser.add_argument('--adv_eps', default=adv_eps, type=float, nargs='+', help='adv_eps')
    parser.add_argument('--ckpt_dir', default=r"D:\cjh\Adversarial_Robustness\ckpt", type=str, help='ckpt_dir')
    parser.add_argument('--timestep', default=timestep, type=int, nargs='+', help='how many timestep to eval')
    parser.add_argument('--samples_num', default=1000, type=int, help='how many samples to eval')
    parser.add_argument('--model_name', default="MobileNetv2",choices=["WideResNet28_10","ResNet18","MobileNetv2","ShuffleNetv2","ResNet"] ,type=str, help='model_name')
    parser.add_argument('--datasets', default="CIFAR10", type=str, help='datasets')

    args = parser.parse_args()

    main(args)
    robust_acc_timestep = np.array(robust_acc_timestep)
    np.save("./robust_acc_{}_{}.npy".format(args.model_name,args.datasets),robust_acc_timestep)