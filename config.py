import os

#项目根目录
root_path = r'D:\cjh\Mask_Generator/'
#存放数据集
datasets_dir = root_path + r'/datasets/'
#存放ImageNet数据集
imageNet_dir = datasets_dir + r'/ImageNet/'
#存放ImageNet对抗样本数据集
ADV_imageNet_dir = datasets_dir + r'/generated/'
#存放模型权重
model_weight_dir = root_path + r'\weight/pretrained/'

save_generator_weight = root_path + r'\weight/generator/'
log_path = root_path + r'\log/'

# 每过多少epoch保存一次
save_epoch_step = 5
ADV_NUM = 10000
shuffle_training_dataset = True
batch_size = 8
train_batch_size = batch_size
validation_batch_size = 4
test_batch_size = 2
generate_data_batch_size = 64
b = 240
c = 0
d = 2*0
epochs = 25*5+1
num_workers = 2
training_lr = 1e-3
training_weight_decay = 1e-4
optim_name = "Adam"

validation_split = .1
test_split = .1
shuffle_dataset = True
random_seed= 42

# crossEntropy could also be used
# the state of MSE loss
SOFT_LABEL = 1 # state of the loss used. mse of logits
HARD_LABEL = 0 # state of the loss used. mse of onehot
loss1_state = SOFT_LABEL
loss2_state = SOFT_LABEL


