# Loss
import torch
from torch.autograd import Variable

loss_MSE = torch.nn.MSELoss()


def MidLayerVectorLoss(femap1,femap2):
    tensor_vector1, tensor_vector2 = getMidLayerVector(femap1, femap2)
    return loss_MSE(tensor_vector1, tensor_vector2)


def getMidLayerVector(femap1,femap2):
    avg_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
    flatten =torch.nn.Flatten()
    tensor_vector1 = torch.ones((femap1[0].shape[0], 0)).cuda()
    tensor_vector2 = torch.ones((femap2[0].shape[0], 0)).cuda()
    for fe1,fe2 in zip(femap1,femap2):
        vector1 = flatten(avg_pooling(fe1))
        vector2 = flatten(avg_pooling(fe2))
        tensor_vector1 = torch.cat([tensor_vector1, vector1], 1)
        tensor_vector2 = torch.cat([tensor_vector2, vector2], 1)

    return tensor_vector1, tensor_vector2


def BoundaryLoss(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(30).type(torch.cuda.FloatTensor)[target.long()].cuda())

    real = torch.sum(target_one_hot * logits[:,:30], 1)
    other = torch.max((1 - target_one_hot) * logits[:,:30] - (target_one_hot * 30), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    return torch.sum(torch.max(other - real, kappa))

# Loss
def Model_Loss(masked_logits,logits, target, kappa=-0., tar=False):
    loss_CE = torch.nn.CrossEntropyLoss()
    loss1 = loss_CE(masked_logits,target)
    loss2 = loss_CE(logits,target)
    loss = 0 if loss1<loss2 else loss1
    return loss


def binary_loss(x):
    x = torch.exp(x-0.5)
    x = torch.pow(x,2)
    return x

