import torch

root_path = r"D:\cjh\Adversarial_Robustness\third_party\fixing_data\ckpt"
model_ckpt_path = {
    "CIFAR10":{
        "ResNet18":{
            "Rebuffi":{
                "l2":r"\cifar10_l2_resnet18_cutmix_ddpm_Rebuffi_21.pt",
                "linf":r"\cifar100_linf_resnet18_ddpm_Rebuffi _21.pt"
            },
            "Gowal":{
                "linf":r"\cifar10_linf_resnet18_ddpm_100m_Gowal_21.pt"
            }
        },
        "WRN28-10":{
            "Rebuffi": {
                "l2":r"\cifar10_l2_wrn28-10_cutmix_ddpm_v2_Rebuffi_21.pt",
                "linf":r"\cifar10_linf_wrn28-10_cutmix_ddpm_v2_Rebuffi_21.pt"
            },
            "Gowal": {
                "linf":r"\cifar10_linf_wrn28-10_ddpm_100m_Gowal_21.pt"
            }
        }
    },
    "CIFAR100":{
        "ResNet18": {
            "Rebuffi": {
                "linf":r"\cifar100_linf_resnet18_ddpm_Rebuffi _21.pt"
            },
            "Gowal": {

            }
        },
        "WRN28-10": {
            "Rebuffi": {
                "linf":r"\cifar100_linf_wrn28-10_cutmix_ddpm_Rebuffi_21.pt"
            },
            "Gowal": {

            }
        }
    }
}


def get_model(datasets,model_name,author):
    from third_party.models import resnet18,WideResNet
    n_classes = 10 if datasets == "CIFAR10" else 100
    if model_name == "WRN28-10":

        model = WideResNet(num_classes=n_classes)

    elif model_name == "ResNet18":

        model = resnet18(num_classes=n_classes)

    ckpt_path = root_path + model_ckpt_path[datasets][model_name][author]['linf']
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt,strict=False)
    model.cuda()
    model.eval()

if __name__ == "__main__":

    model = get_model("CIFAR10","WRN28-10","Rebuffi")

        