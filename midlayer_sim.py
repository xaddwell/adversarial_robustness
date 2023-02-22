from utils.midlayer_sim import draw_2midlayer_sim
from utils.get_models import get_trained_classifier,get_trained_generator
from utils.get_trainingloader import get_loader
import matplotlib.pyplot as plt
import torch
import numpy as np

def test_for_single(test_model_name,
                    test_attack_method,
                    generator = None):

    test_model = get_trained_classifier(test_model_name,feature_map=True).cuda()
    test_model_name = test_model_name.split("_")[0]
    test_model.eval()
    if generator!=None:
        generator = get_trained_generator(test_model_name,test_attack_method,"ResUNet01").cuda()
        generator.eval()
    _, _, test_loader = get_loader(test_model_name,
                                   test_attack_method,
                                   test_batch_size=1,
                                   test_split=0.01)

    AEs_vs_NEs = []
    NEs_vs_NEsG = []
    AEs_vs_MAEs = []
    NEs_vs_MNEs = []
    MAEs_vs_NEs = []
    sum_ori = 0
    sum_advs = 0
    sum_oris_noise = 0
    sum_ori_add_mask = 0
    sum_advs_add_mask = 0
    sum_num = 0

    for iter,(oris,advs,labels) in enumerate(test_loader):

        oris = oris.cuda()
        advs = advs.cuda()

        idx = torch.randperm(oris.nelement())
        oris_noise = (advs-oris).view(-1)[idx].view(oris.size())
        oris_noise = torch.clip(oris + oris_noise.cuda(), min=0, max=1)
        labels = labels.cuda()

        if generator != None:
            advs_add_mask = (torch.tanh(generator(advs) - advs) + 1) / 2
            ori_add_mask = (torch.tanh(generator(oris) - oris) + 1) / 2
            logits_ori_add_mask, fea_ori_add_mask = test_model(ori_add_mask)
            logits_advs_add_mask, fea_advs_add_mask = test_model(advs_add_mask)

            pred_ori_add_mask= logits_ori_add_mask.argmax(dim=1)
            pred_advs_add_mask = logits_advs_add_mask.argmax(dim=1)

            temp_ori_add_mask = torch.sum(pred_ori_add_mask == labels)
            temp_advs_add_mask = torch.sum(pred_advs_add_mask == labels)

            sum_ori_add_mask+= temp_ori_add_mask
            sum_advs_add_mask += temp_advs_add_mask


        logits_ori, fea_ori = test_model(oris)
        logits_advs, fea_advs = test_model(advs)
        logits_oris_noise, fea_ori_noise = test_model(oris_noise)

        pred_ori = logits_ori.argmax(dim=1)
        pred_advs = logits_advs.argmax(dim=1)
        pred_oris_noise = logits_oris_noise.argmax(dim=1)

        temp_ori = torch.sum(pred_ori == labels)
        temp_advs = torch.sum(pred_advs == labels)
        temp_oris_noise = torch.sum(pred_oris_noise == labels)

        sum_ori += temp_ori
        sum_advs += temp_advs
        sum_oris_noise += temp_oris_noise
        sum_num += len(labels)

        if temp_ori == 1 and temp_advs == 0 and temp_oris_noise == 1 :
            temp1 = []
            temp2 = []
            for i in range(len(fea_ori)):
                temp1.append(cos(fea_ori[i],fea_advs[i]))
                temp2.append(cos(fea_ori[i],fea_ori_noise[i]))
            if generator != None:
                if temp_ori_add_mask == 1 and \
                   temp_advs_add_mask == 1 and \
                   temp_advs == 0:
                    temp3 = []
                    temp4 = []
                    temp5 = []
                    for j in range(len(fea_ori)):
                        temp3.append(cos(fea_ori_add_mask[j], fea_ori[j]))
                        temp4.append(cos(fea_advs_add_mask[j], fea_advs[j]))
                        temp5.append(cos(fea_ori[j], fea_advs_add_mask[j]))

                    AEs_vs_MAEs.append(np.array(temp4))
                    NEs_vs_MNEs.append(np.array(temp3))
                    MAEs_vs_NEs.append(np.array(temp5))
                    AEs_vs_NEs.append(np.array(temp1))
                    NEs_vs_NEsG.append(np.array(temp2))
            else:
                AEs_vs_NEs.append(np.array(temp1))
                NEs_vs_NEsG.append(np.array(temp2))


    draw_2midlayer_sim(name="AEs_vs_NEs",sim_list = \
        np.array(AEs_vs_NEs)[np.newaxis,:],color="red",normalization=False,num=0.4)
    draw_2midlayer_sim(name="NEs_vs_NEsG",sim_list = \
        np.array(NEs_vs_NEsG)[np.newaxis,:], color="blue",normalization=False,num=0.4)

    if generator!=None:
        draw_2midlayer_sim(name="AEs_vs_MAEs", sim_list= \
            np.array(AEs_vs_MAEs)[np.newaxis, :], color="yellow", normalization=False,num=0.4)
        draw_2midlayer_sim(name="NEs_vs_MNEs", sim_list= \
            np.array(NEs_vs_MNEs)[np.newaxis, :], color="green", normalization=False,num=0.4)
        draw_2midlayer_sim(name="MAEs_vs_NEs", sim_list= \
            np.array(MAEs_vs_NEs)[np.newaxis, :], color="black", normalization=False,num=0.4)

    plt.title(test_attack_method+"_atk_"+test_model_name)
    plt.savefig("visualization/midlayer_diff/"+\
                test_attack_method+"_atk_"+\
                test_model_name+".jpg",dpi=800)
    plt.show()

    print("total: {} samples, ori:{:.3f} advs:{:.3f} oris_noise:{:.3f}".\
        format(sum_num, sum_ori/sum_num,sum_advs/sum_num,sum_oris_noise/sum_num))


if __name__ == "__main__":
    victim_model_list = ["ShuffleNetv2"]
    attack_method_list = ["PGD","FGSM","DIFGSM","MIFGSM"]
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-5)

    for victim_model_name in victim_model_list:
        for attack_method in attack_method_list:
            test_for_single(victim_model_name+"_with_allfea",
                            attack_method,generator = "ResUNet01")