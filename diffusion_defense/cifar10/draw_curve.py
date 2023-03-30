import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plt_acc(file_path):
    plt.style.use('ggplot')
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    names = ["TimeSteps", "Clean Accuracy", "Robust Accuracy"]

    robust_acc = np.array(np.load(file_path),dtype=np.float32).transpose()
    x_axis = robust_acc[0]
    y_axis = robust_acc[1:3]
    plt.plot(x_axis,y_axis[0]*100,label = names[1],marker = "o")
    plt.plot(x_axis, y_axis[1]*100,label = names[2],marker = "s")
    plt.xlabel(names[0])
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("./TimeStep-Acc.png",dpi=800)
    plt.show()

def plt_robust_radius(file_paths):
    plt.style.use('ggplot')
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    marker_list = ['s','o','v','*']
    for i,file_path in enumerate(file_paths):
        name = file_path.split('_')[-2]
        robust_acc = np.array(np.load(file_path),dtype=np.float32).transpose()
        x_axis = robust_acc[-1]
        y_axis = robust_acc[-2]
        plt.plot(x_axis,y_axis*100,label = name ,marker = marker_list[i])
    plt.xlabel("Perturbation Radius")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("./Radius-Acc.png",dpi=800)
    plt.show()


if __name__ == "__main__":
    # file_path = r"./robust_acc_timestep.npy"
    # plt_acc(file_path)

    file_paths = [r"./robust_acc_MobileNetv2_CIFAR10.npy",
                 r"./robust_acc_ResNet18_CIFAR10.npy",
                 r"./robust_acc_ResNet56_CIFAR10.npy",
                 r"./robust_acc_ShuffleNetv2_CIFAR10.npy"
                 ]
    plt_robust_radius(file_paths)