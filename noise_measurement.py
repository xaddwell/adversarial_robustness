import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from config import *
from PIL import Image
from torchvision import transforms

to_tensor = transforms.ToTensor()

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

def yCbCr2rgb(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[1.164, 1.164, 1.164],
                        [0, -0.392, 2.017],
                        [1.596, -0.813, 0]])
    bias = torch.tensor([-16.0 / 255.0, -128.0 / 255.0, -128.0 / 255.0])
    temp = (im_flat + bias).mm(mat)
    out = temp.view(3, list(input_im.size())[1], list(input_im.size())[2])
    return out


def rgb2yCbCr(input_im):
    im_flat = input_im.contiguous().view(-1, 3).float()
    mat = torch.tensor([[0.257, -0.148, 0.439],
                        [0.564, -0.291, -0.368],
                        [0.098, 0.439, -0.071]])
    bias = torch.tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0])
    temp = im_flat.mm(mat) + bias
    out = temp.view(3, input_im.shape[1], input_im.shape[2])
    return out

# 计算PSNR
def PSNR(target, pred, R=1, dummy=1e-4, reduction='mean'):
    target = target.squeeze(0)
    pred = pred.squeeze(0)
    target = rgb2yCbCr(target)[0]
    pred = rgb2yCbCr(pred)[0]
    with torch.no_grad():
        dims = (1, 2, 3) if len(target.shape) == 4 else 1
        mean_sq_err = ((target - pred) ** 2).mean(dims)
        mean_sq_err = mean_sq_err + (mean_sq_err == 0).float() * dummy  # if 0, fill with dummy -> PSNR of 40 by default
        output = 10 * torch.log10(R ** 2 / mean_sq_err)
        if reduction == 'mean':
            return output.mean()
        elif reduction == 'none':
            return output

if __name__ == "__main__":

    victim_model = 'ShuffleNetv2'
    source_attack = 'DIFGSM'
    target_model = 'ShuffleNetv2'
    target_attack = 'DIFGSM'
    path = root_path + r'\visualization\mask-ori-advs\{}\{}\{}\{}/'
    legend = np.array([['ori','ori_add_mask','advs','advs_add_mask']])
    path = path.format(victim_model,source_attack,target_model,target_attack)
    pic_dict = {}
    for f in os.listdir(path):
        fname = f.split('_')[0]
        if fname not in pic_dict.keys():
            pic_dict[fname] = path + str(fname) + r'_{}.jpg'

    psnr_mat = torch.zeros((4, 4))
    ssim_mat = torch.zeros((4, 4))

    for k,v in pic_dict.items():

        ori = to_tensor(Image.open(v.format('ori'))).unsqueeze(0)
        ori_add_mask = to_tensor(Image.open(v.format('ori_add_mask'))).unsqueeze(0)
        advs = to_tensor(Image.open(v.format('advs'))).unsqueeze(0)
        advs_add_mask = to_tensor(Image.open(v.format('advs_add_mask'))).unsqueeze(0)
        pics = [ori,ori_add_mask,advs,advs_add_mask]

        for i,a in enumerate(pics):
            for j,b in enumerate(pics):
                psnr_mat[i][j] += float(PSNR(a,b))
                ssim_mat[i][j] += float(ssim(a, b))

    print('PSNR:')
    print(np.concatenate((legend,
                          np.array(psnr_mat)/len(pic_dict)),axis=0))

    print('SSIM:')
    print(np.concatenate((legend,
                          np.array(ssim_mat)/len(pic_dict)),axis=0))







