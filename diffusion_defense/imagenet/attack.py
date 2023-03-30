import argparse
import time
import torch
import datetime
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from core import Smooth
from imagenet.DRM import DiffusionRobustModel
import torchattacks as ta
import os
import sys
import numpy as np
sys.path.append("D:\cjh\Adversarial_Robustness")
from utils.utils import get_classifier, model_load_ckpt_eval
from utils.transform import cifar10_raw2clf


ImageNet_DATA_DIR = r"D:\cjh\Adversarial_Robustness\datasets\ImageNet"


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

    img_size = 256
    classifier = model_load_ckpt_eval(args, get_classifier(args, pretrained=False))
    model = DiffusionRobustModel(classifier.cuda())
    dataset = datasets.ImageFolder(ImageNet_DATA_DIR, transform=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()]))
    indices = range(args.samples_num)
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)

    pert_list = args.adv_eps

    def save_fig(x,x_adv,t):
        x_denoise = model.denoise(x,t)
        x_denoise = torch.nn.functional.interpolate(x_denoise, (224, 224))
        y = classifier(x).argmax(dim=1)
        y_adv = classifier(x_adv).argmax(dim=1)
        y_denoise = classifier(x_denoise).argmax(dim=1)
        path_list = r"D:\cjh\Adversarial_Robustness\diffusion_defense\imagenet\imgs/"
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



    for pert in pert_list:

        args.adv_eps = pert
        attacker = get_attack(args,_method="PGD",target_model=classifier)

        for timestep in args.timestep:

            use_cond = True;multistep = False
            total_num = 0;correct = 0;correct_adv = 0;correct_d_adv = 0;correct_d = 0
            for x,y in loader:

                x = x.cuda()
                y = y.cuda()

                x_adv = attacker(x,y)

                # save_fig(x,x_adv,timestep)

                pred = classifier(x).argmax(dim=1)
                pred_d = model(x, timestep).argmax(dim=1)
                pred_adv = classifier(x_adv).argmax(dim=1)
                pred_d_adv = model(x_adv, timestep).argmax(dim=1)


                correct += (pred == y).sum().cpu()
                correct_d += (pred_d == y).sum().cpu()
                correct_adv += (pred_adv == y).sum().cpu()
                correct_d_adv += (pred_d_adv == y).sum().cpu()


                total_num += len(y)

            # robust_acc_timestep.append([timestep,correct_c/float(total_num),correct_c_adv/float(total_num),pert])

                print("Adv_eps: {} TimeStep:{} Diffusion: Std_acc {:.3f} Robust_acc {:.3f} || Classifier: Std_acc {:.3f} Robust_acc {:.3f} ".
                      format(pert,timestep,correct_d/float(total_num),correct_d_adv/float(total_num),
                             correct/float(total_num),correct_adv/float(total_num)))

if __name__ == "__main__":

    timestep = [i for i in range(50,500,10)]
    adv_eps = [i/255 for i in range(8,9,1)]
    robust_acc_timestep = []

    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    parser.add_argument('--steps', default=5, type=int, help='adv_steps')
    parser.add_argument('--adv_eps', default=adv_eps, type=float, nargs='+', help='adv_eps')
    parser.add_argument('--ckpt_dir', default=r"D:\cjh\Adversarial_Robustness\ckpt", type=str, help='ckpt_dir')
    parser.add_argument('--timestep', default=timestep, type=int, nargs='+', help='how many timestep to eval')
    parser.add_argument('--samples_num', default=1000, type=int, help='how many samples to eval')
    parser.add_argument('--model_name', default="MobileNetv2",choices=["Inception-ResNet-v2","ResNet18","MobileNetv2","ShuffleNetv2","SwinT-small"] ,type=str, help='model_name')
    parser.add_argument('--datasets', default="ImageNet", type=str, help='datasets')

    args = parser.parse_args()

    main(args)