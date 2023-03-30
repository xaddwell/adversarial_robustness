import torch
import torch.nn as nn
import pytorch_ssim

import os
import sys

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

sys.path.append("D:\cjh\Adversarial_Robustness")

from utils.transform import diff2raw,raw2diff
from utils.metrix import SSIM

diffusion_ckpt_path = r"D:\cjh\Adversarial_Robustness\ckpt\diffusion\cifar10_uncond_50M_500K.pt"
MSE = nn.MSELoss()
ssim = SSIM()

class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionRobustModel(nn.Module):
    def __init__(self,classifier):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load(diffusion_ckpt_path)
        )
        model.eval().cuda()

        self.model = model 
        self.diffusion = diffusion 

        classifier.eval().cuda()
        self.classifier = classifier

    def forward(self, x, t):
        x_in = x * 2 -1
        imgs = diff2raw(self.denoise(x_in, t))
        with torch.no_grad():
            out = self.classifier(imgs)
        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda()

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out


class DistributionDiffusion(nn.Module):
    def __init__(self,classifier):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load(diffusion_ckpt_path)
        )
        model.eval().cuda()

        self.model = model
        self.diffusion = diffusion

        classifier.eval().cuda()
        self.classifier = classifier

    def forward(self, x, t, multistep=False,use_cond = False):
        x_in = x * 2 -1
        imgs = diff2raw(self.denoise(x_in, t, multistep,use_cond))
        with torch.no_grad():
            out = self.classifier(imgs)
        return out

    def calParams(self,t):
        mu = 1/t
        scale = mu
        return mu,scale

    def denoise(self, x_start, t, multistep=False,use_cond = False):

        t_batch = torch.tensor([t] * len(x_start)).cuda()
        noise = torch.randn_like(x_start)
        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        def cond_fn(x_reverse_t, t):
            with torch.enable_grad():
                x_0 = x_reverse_t.detach().requires_grad_(True)
                x_t_1 = self.diffusion.q_sample(x_start=x_0, t=torch.tensor([1] * len(x_start)).cuda(), noise=noise)
                mu,scale = self.calParams(100)

                loss1 = MSE(self.classifier(x_0),self.classifier(x_start))
                grad1 = torch.autograd.grad(loss1, x_0)[0]
                loss2 = MSE(self.classifier(x_0), self.classifier(x_t_1))
                grad2 = torch.autograd.grad(loss2, x_0)[0]
                loss3 = 0*MSE(x_0, x_t_start)
                grad3 = torch.autograd.grad(loss3, x_0)[0]
                # print("timestep {} loss1:{:.3f} loss2:{:.3f} loss3:{:.3f} ".format(t[0],loss1,loss2,loss3))
                return -(grad3 * scale + mu*grad1 + grad2)


        with torch.no_grad():
            if not use_cond:
                cond_fn = None

            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    t_batch = torch.tensor([i] * len(x_start)).cuda()
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True,
                        cond_fn = cond_fn
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True,
                    cond_fn = cond_fn
                )['pred_xstart']

        return out