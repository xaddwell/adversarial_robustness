import os
from omegaconf import OmegaConf

cfg = OmegaConf.create({

    "root_dir": r'D:\cjh\Adversarial_Robustness/',
    "datasets_dir": "${root_dir}" + 'datasets/',
    "ckpt_dir": "${root_dir}" + 'ckpt/',
    "log_dir": "${root_dir}" + 'result/',
    "result_dir": "${root_dir}" + 'result/',

    "optimizer": "Adam",
    "lr": 1e-3,
    "weight_decay": 1e-4,

    "max_epochs": 100,
    "num_workers": 2,
    "batch_size": 8,
    "save_ckpt_interval_epoch": 5,

    "val_split": .1,
    "test_split": .1,
    "shuffle_dataset": True,
    "random_seed": 42
})


__all__ = ['get_default_cfg']

def get_default_cfg():

    return cfg