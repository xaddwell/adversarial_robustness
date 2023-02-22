import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
os.environ['KMP_DUPLICATE_LIB_OK']='True'
base_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize([224,224])])

to_tensor = transforms.ToTensor()
_resize = transforms.Resize([224,224])

if __name__ == '__main__':
    base_dir = r'D:\cjh\Mask_Generator\verify_backdoor\data'
    bg_dir = os.path.join(base_dir,'bg')
    bg_id = ['cd','hd','pp','tk','zt','sy']
    ori_dir = os.path.join(base_dir,'segdata','output')
    atk_dir = os.path.join(base_dir,'atk_data')
    for d1 in os.listdir(bg_dir):
        if not os.path.exists(os.path.join(atk_dir,d1)):
            os.mkdir(os.path.join(atk_dir,d1))
        p1 = os.path.join(bg_dir,d1)
        for d2 in os.listdir(p1):
            p2 = os.path.join(p1,d2)
            img = base_transform(Image.open(p2))
            for d3 in os.listdir(ori_dir):
                p3 = os.path.join(atk_dir,d1,d3)
                if not os.path.exists(p3):
                    os.mkdir(p3)
                for d4 in os.listdir(os.path.join(ori_dir,d3)):
                    if d4.split('.')[1] == 'jpg':
                        id = d4.split('.')[0]
                        mask_dir = str(os.path.join(ori_dir,d3,id)) + '.png'
                        ori_path = str(os.path.join(ori_dir,d3,id)) + '.jpg'
                        mask = torch.ceil(to_tensor(Image.open(mask_dir)))
                        _img = to_tensor(Image.open(ori_path))*mask
                        atk_img = img*_resize(1-mask) + _resize(_img)
                        to_pil_image(atk_img).save(os.path.join(p3,d2.split('.')[0]+'_'+d4))