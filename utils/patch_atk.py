import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from  torchvision import utils as vutils
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms
from PIL import ImageFile
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
ImageFile.LOAD_TRUNCATED_IMAGES = True

from default_config import *



class AdversarialPatch():

    def __init__(self,model,save_path,
                 targeted_atk,
                 patch_type='rectangle',
                 image_size=(3, 224, 224),
                 noise_percentage=0.03):

        self.patch_type = patch_type
        self.noise_percentage = noise_percentage
        self.image_size = image_size
        self.model = model
        self.save_path = save_path
        self.targeted_atk = targeted_atk
        self.initialize_patch()

    def initialize_patch(self):
        if self.patch_type == 'rectangle':
            mask_length = int((self.noise_percentage * self.image_size[1] * self.image_size[2]) ** 0.5)
            patch = np.random.rand(self.image_size[0], mask_length, mask_length)
            print('The shape of the initialized patch is {}'.format(patch.shape))
            self.patch = patch
        else:
            print('Only support rectangle patch at present')


    def optimize_patch(self,train_loader,
                       epochs,lr,max_iteration,
                       probability_threshold,target):

        best_patch_success_rate = 0
        advs_num = 0
        patch =self.patch
        model = self.model
        if self.targeted_atk:
            target = self.targeted_atk
        for epoch in range(epochs):
            train_total, train_actual_total, train_success = 0, 0, 0
            for iter,(image, label) in enumerate(train_loader):
                train_total += label.shape[0]
                assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
                image = image.cuda()
                label = label.cuda()
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
                    train_actual_total += 1
                    applied_patch, mask, x_location, y_location = \
                        self.mask_generation(self.patch_type, patch,image_size=(3, 224, 224))
                    perturbated_image, applied_patch = \
                        self.patch_attack(image, applied_patch,probability_threshold,mask,target)
                    perturbated_image = torch.from_numpy(perturbated_image).cuda()
                    output = model(perturbated_image)
                    _, predicted = torch.max(output.data, 1)
                    if self.targeted_atk and predicted[0].data.cpu().numpy() == target:
                        train_success += 1
                        vutils.save_image(perturbated_image[0].cpu(),self.save_path + "/advs/{}_{}_{}.jpg".format(epoch, iter, label[0]),
                                              normalize=False)  # 保存对抗样本
                        vutils.save_image(image[0].cpu(), self.save_path + "/ori/{}_{}_{}.jpg".format(epoch, iter,label[0]),
                                              normalize=False)  # 保存干净样本

                    patch = applied_patch[0][:, x_location:x_location + patch.shape[1],
                            y_location:y_location + patch.shape[2]]

            if advs_num == 1000:
                break

            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
            plt.show()
            print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".
                  format(epoch,100 * train_success / train_actual_total))
            train_success_rate = self.test_patch(self.patch_type, patch, test_loader)
            print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".
                  format(epoch, 100 * train_success_rate))
            test_success_rate = self.test_patch(self.patch_type, patch, test_loader)
            print("Epoch:{} Patch attack success rate on testset: {:.3f}%".
                  format(epoch, 100 * test_success_rate))

            if test_success_rate > best_patch_success_rate:
                self.patch = patch
                best_patch_success_rate = test_success_rate
                print("Save best patch")
                plt.imshow(np.clip(np.transpose(patch, (1, 2, 0)) * std + mean, 0, 1))
                plt.show()



    def patch_attack(self,image,
                     applied_patch,
                     probability_threshold,
                     mask, target,lr=1,
                     max_iteration=100):

        model = self.model
        model.eval()
        applied_patch = torch.Tensor(applied_patch)
        mask = torch.Tensor(mask)
        target_probability, count = 0, 0
        perturbated_image = torch.mul(mask.type(torch.FloatTensor),applied_patch.type(torch.FloatTensor)) + \
                            torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))

        while target_probability < probability_threshold and count < max_iteration:
            count += 1
            # Optimize the patch
            perturbated_image = Variable(perturbated_image.data, requires_grad=True)
            per_image = perturbated_image
            per_image = per_image.cuda()
            output = model(per_image)

            target_log_softmax = torch.nn.functional.log_softmax(output,dim=1)[0][target]
            torch.nn.LogSoftmax()
            target_log_softmax.backward()

            patch_grad = perturbated_image.grad.clone().cpu()
            perturbated_image.grad.data.zero_()
            if self.targeted_atk:
                applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
            else:
                applied_patch = applied_patch.type(torch.FloatTensor) - lr * patch_grad

            applied_patch = torch.clamp(applied_patch, min=-3, max=3)
            # Test the patch
            perturbated_image = torch.mul(mask.type(torch.FloatTensor),
                                          applied_patch.type(torch.FloatTensor)) + torch.mul(
                (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            if self.targeted_atk:
                target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]
            else:
                temp_pred = torch.nn.functional.softmax(output, dim=1).data[0]
                temp_pred[target] = 0
                target_probability = torch.max(temp_pred)

        perturbated_image = perturbated_image.cpu().numpy()
        applied_patch = applied_patch.cpu().numpy()
        return perturbated_image, applied_patch

    def test_patch(self,patch_type, patch, test_loader):
        """
        :param patch_type:
        :param target:
        :param test_loader:
        :param model:
        :return:
        """
        model = self.model
        model.eval()
        test_total, test_actual_total, test_success = 0, 0, 0
        for (image, label) in test_loader:
            test_total += label.shape[0]
            assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
            image = image.cuda()
            label = label.cuda()
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0] == label:

                test_actual_total += 1
                applied_patch, mask, x_location, y_location = \
                    self.mask_generation(patch_type, patch,image_size=(3, 224, 224))
                applied_patch = torch.from_numpy(applied_patch)
                mask = torch.from_numpy(mask)
                perturbated_image = torch.mul(mask.type(torch.FloatTensor),
                                              applied_patch.type(torch.FloatTensor)) + torch.mul(
                    (1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
                perturbated_image = perturbated_image.cuda()

                output = model(perturbated_image)
                _, predicted = torch.max(output.data, 1)

                if self.targeted_atk and predicted[0].data.cpu().numpy() == self.targeted_atk:
                    test_success += 1
                elif self.targeted_atk == False and predicted[0]!= label:
                    test_success += 1

        return test_success / test_actual_total

    def mask_generation(self,mask_type='rectangle',
                        patch=None,
                        image_size=(3, 224, 224)):

        applied_patch = np.zeros(image_size)
        if mask_type == 'rectangle':
            # patch rotation
            rotation_angle = np.random.choice(4)
            for i in range(patch.shape[0]):
                patch[i] = np.rot90(patch[i], rotation_angle)  # The actual rotation angle is rotation_angle * 90
            # patch location
            x_location, y_location = np.random.randint(low=0, high=image_size[1] - patch.shape[1]), np.random.randint(
                low=0, high=image_size[2] - patch.shape[2])
            for i in range(patch.shape[0]):
                applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch
        mask = np.array(applied_patch.copy())
        mask[mask != 0] = 1.0
        return applied_patch, mask, x_location, y_location


    def atk(self,img):
        pass





if __name__=="__main__":
    patch_atk_config = {
        "train_size": 2000,
        "test_size": 1000,
        "noise_percentage": 0.16,
        "probability_threshold": 0.9,
        "lr": 1.0,
        "max_iteration": 100,
        "target_label": None,
        "epochs": 20,
        "patch_type": "rectangle",
        "targeted_atk":2
    }

    temp = AdversarialPatch(model=model,save_path = path,
                            targeted_atk = targeted_atk,
                            patch_type = patch_type,
                            image_size=(3,224,224),
                            noise_percentage=noise_percentage)
    temp.initialize_patch()
    temp.optimize_patch(train_loader=train_loader,
                        target=targeted_atk,
                        epochs=epochs,lr=lr,
                        max_iteration=max_iteration,
                        probability_threshold=probability_threshold)
