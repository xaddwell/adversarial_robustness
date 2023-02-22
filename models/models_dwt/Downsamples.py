from models.ImageNet.DWT_IDWT import *
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.dwt = DWT_2D_tiny(wavename = wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL