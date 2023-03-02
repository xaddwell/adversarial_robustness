
from utils.utils import get_classifier,get_logger
from utils.datasets import get_loader
from utils.transform import get_transform
import torch
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_optimizer(name,parameters,lr,weight_decay):
    if name == "Adam":
        optimizer = torch.optim.Adam(parameters,
                                      lr=lr,
                                      weight_decay=weight_decay)
    elif name == "SGD":
        optimizer = torch.optim.SGD(parameters,
                                    lr=lr,
                                    weight_decay=weight_decay)
    else:
        return None

    return optimizer

class Trainer(object):
    def __init__(self, args):
        if args.resume:
            ckpt_name = "{}_{}_ckpt_best.pth".format(args.datasets, args.model_name)
            path_checkpoint = os.path.join(args.ckpt_dir,"pretrained/" + ckpt_name)  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点

            self.model = get_classifier(args, pretrained=False)
            self.model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            if args.use_cuda:
                self.model = self.model.cuda()

            self.optimizer = set_optimizer(args.optimizer, self.model.parameters(), args.lr, args.weight_decay)
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            self.start_epochs = checkpoint['epoch']  # 设置开始的epoch

        else:

            self.start_epochs = 0
            self.model = get_classifier(args = args, pretrained=True)
            self.optimizer = set_optimizer(args.optimizer, self.model.parameters(), args.lr, args.weight_decay)

            if args.use_cuda:
                self.model = self.model.cuda()

        transform = get_transform(args.model_name,args.datasets,stage="train")
        train_loader, val_loader = get_loader(datasets_name=args.datasets,stage='train',
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              transform=transform)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_cuda = args.use_cuda
        self.args = args
        self.max_epochs = args.max_epochs
        self.iteration = 0
        self.best_acc = 0
        self.logger = get_logger(args.log_dir)

    def run(self):
        for i in range(self.start_epochs,self.max_epochs):
            loss, acc = self.train()
            self.logger.print('[Train] Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(i, self.max_epochs, loss, acc))
            self.iteration = i
            if i % self.args.save_ckpt_interval_epoch == 0:
                acc = self.evaluation()
                if acc > self.best_acc:
                    self.best_acc = acc
                    print("[Val] Epoch:[{}/{}]\tbest_acc={:.3f}, ckpt saved".format(i, self.max_epochs,acc))
                    self.save_ckpt(is_best=True)
                else:
                    self.save_ckpt(is_best=False)

    def train(self):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        epoch_sum = 0

        for i, (x, y) in enumerate(self.train_loader):
            if self.use_cuda:
                x = x.cuda()
                y = y.cuda()

            out = self.model(x)
            loss = self.criterion(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_acc += (torch.max(out, 1)[1] == y).sum()
            epoch_loss += loss
            epoch_sum += len(y)

        return epoch_loss / epoch_sum , epoch_acc / epoch_sum

    def evaluation(self):
        self.model.eval()
        with torch.no_grad(): # 或者@torch.no_grad() 被他们包裹的代码块不需要计算梯度， 也不需要反向传播
            eval_acc = 0
            eval_sum = 0
            for i, (x, y) in enumerate(self.val_loader):
                if self.use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                out = self.model(x)
                prediction = torch.max(out, 1)[1]
                pred_correct = (prediction == y).sum()
                eval_acc += pred_correct.item()
                eval_sum += len(y)
            return eval_acc / eval_sum

    def save_ckpt(self, is_best):
        checkpoint = {
            "net": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": self.iteration
        }
        if is_best:
            ckpt_name = "/{}_{}_ckpt_best.pth".format(self.args.datasets,self.args.model_name)
            torch.save(checkpoint, (self.args.ckpt_dir + "/pretrained/" + ckpt_name))
        else:
            ckpt_name = "/{}_{}_ckpt.pth".format(self.args.datasets, self.args.model_name,self.iteration)
            torch.save(checkpoint, (self.args.result_dir + ckpt_name))
