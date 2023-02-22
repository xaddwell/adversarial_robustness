import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),#参数是padding，使用输入 tensor的反射来填充
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        #torch.nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
                        #Applies Instance Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper Instance Normalization: The Missing Ingredient for Fast Stylization .
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]
        #torch.nn.Sequential(*args)
        #A sequential container.模型将按照在构造函数中传递的顺序添加到模型中。 或者，也可以传递模型的有序字典。
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # 初始卷积块
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
#ReflectionPad2d（）搭配7x7卷积，先在特征图周围以反射的方式补长度，使得卷积后特征图尺寸不变
#InstanceNorm2d（）是相比于batchNorm更加适合图像生成，风格迁移的归一化方法，相比于batchNorm跨样本，
#单通道统计，InstanceNorm采用单样本，单通道统计，括号中的参数代表通道数
        # 下采样
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        #残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # 上采样
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # 输出层
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ] #nn.Tanh()：Applies the element-wise function:

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

