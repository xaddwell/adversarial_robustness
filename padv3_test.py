import torch.nn as nn
import torch
import numpy as np
import random

def grad_pytorch_impl_fp16():
    out_grad_np_copy = ()
    x.requires_grad = True
    paddings = tuple(input_paddings)
    for idx in range(len(out_grad_np)):
        if (dtype == np.complex64) or (dtype == np.complex128) or (dtype == np.float16):
            out_grad_np_copy += (out_grad_np[idx],)
        elif (dtype == np.uint8) or (dtype == np.uint16) or (dtype == np.uint32) or (dtype == np.uint64):
            out_grad_np_copy += (out_grad_np,)
        else:
            out_grad_np_copy += (out_grad_np,)
    net = nn.ReplicationPad2d(paddings).cuda()
    out = net(x).cuda()
    for i in range(len(out)) :
        if i != len(out)-1:
            out[i].backward(gradient=torch.tensor(out_grad_np_copy[i]), retain_graph=True)
        else:
            out[i].backward(gradient=torch.tensor(out_grad_np_copy[i]))
    return x.grad.cpu().numpy().astype(np.float16)

seed = 42
# 生成随机数，以便固定后续随机数，方便复现代码
random.seed(seed)
# 没有使用GPU的时候设置的固定生成的随机数
np.random.seed(seed)
# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.manual_seed(seed)
# torch.cuda.manual_seed()为当前GPU设置随机种子
torch.cuda.manual_seed(seed)

out_grad_np = torch.tensor([[[[-7.3547e-02,-1.8652e+00,-2.9922e+00,-1.1885e+00, 1.4319e-01,-8.3838e-01,1.3291e+00,-8.9844e-01,-1.7764e+00],
[1.0039e+00,-9.0039e-01,-1.4715e-03,4.9219e-01,-2.4695e-01,-3.2422e-01,-2.3120e-01,-1.2119e+00,9.0820e-01],
[-1.2537e-01,7.9102e-01,-1.5068e+00,-1.0332e+00,1.6543e+00,1.5986e+00,-1.3428e+00,-2.9321e-01,-5.0635e-01],
[-2.1729e-01,1.3504e-02,2.8262e+00,2.3999e-01,3.0945e-02,8.0322e-01,1.0234e+00,7.6318e-01,-9.4385e-01],
[1.5635e+00,-3.6841e-01,8.3008e-01,5.3125e-01,1.9902e+00,-1.1699e+00,1.4417e-01,-4.6558e-01,8.9453e-01],
[1.4414e+00,5.9277e-01,-7.1484e-01,-9.7852e-01,-1.0352e+00,1.7358e-01,3.8721e-01,-7.2656e-01,-1.0840e+00],
[-6.0254e-01,-2.5122e-01,2.6001e-02,2.1562e+00,1.8542e-01,-7.3096e-01,-1.5889e+00,-2.8833e-01,8.2568e-01],
[4.2749e-01,1.2773e+00,1.2744e-01,-3.9429e-01,9.7021e-01,1.2537e-01,-5.0928e-01,-7.4121e-01,-1.0615e+00],
[1.2998e+00,-9.5166e-01,-1.2783e+00,8.3154e-01,5.2826e-02,-1.7529e-01,4.6021e-01,-4.7192e-01,-1.0791e+00],
[-4.8340e-01,-3.1885e-01,1.3196e-01,1.8262e+00,-5.9473e-01,7.1387e-01,-2.5342e-01,-6.3037e-01,-8.3740e-01]]]],dtype=torch.float16).cuda()
x = torch.randn((1,1,3,3),dtype=torch.float16).cuda()
input_paddings = (2,4,3,4)
dtype=np.float16
print("seed=",seed)
print("out_grad_np=",out_grad_np)
print("grad=",grad_pytorch_impl_fp16())
