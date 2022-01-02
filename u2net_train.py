import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import StylObjDataset

from model import U2NET
from model import U2NETP

if __name__ == '__main__':

    # ------- 1. define loss function --------

    bce_loss = nn.BCELoss(size_average=True)

    def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, styles_v):

        loss0 = bce_loss(d0,styles_v)
        loss1 = bce_loss(d1,styles_v)
        loss2 = bce_loss(d2,styles_v)
        loss3 = bce_loss(d3,styles_v)
        loss4 = bce_loss(d4,styles_v)
        loss5 = bce_loss(d5,styles_v)
        loss6 = bce_loss(d6,styles_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        #print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

        return loss0, loss


    # ------- 2. set the directory of training dataset --------

    model_name = 'u2netp' #'u2net'
    #model_name = 'u2net' #'u2netp'

    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('ffhq' + os.sep)
    tra_style_dir = os.path.join('comics_heroes' + os.sep)

    image_ext = '.jpg'
    style_ext = '.jpg'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)

    if os.path.exists(model_dir + model_name) == False:
        os.mkdir(model_dir + model_name)

    epoch_num = 100
    batch_size_train = 12
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    tra_lbl_name_list = glob.glob(data_dir + tra_style_dir + '*' + image_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train styles: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    stylobj_dataset = StylObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(512),
            RandomCrop(288),
            ToTensorLab(flag=0)]))

    stylobj_dataloader = DataLoader(stylobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 3)
    elif(model_name=='u2netp'):
        net = U2NETP(3,3)

    chkpnt_name = 'u2netp_bce_itr_8000_train_4.034824_tar_0.572258'
    chkpnt_dir = os.path.join(os.getcwd(), 'saved_models', model_name, chkpnt_name + '.pth')
    net.load_state_dict(torch.load(chkpnt_dir))

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    # ------- 5. training process --------
    print("---start training...")
    ite_num = 8001
    start_epoch = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 2000 # save the model every 2000 iterations
    print_freq = 50

    for epoch in range(start_epoch, epoch_num):
        net.train()

        for i, data in enumerate(stylobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, styles = data['image'], data['style']

            inputs = inputs.type(torch.FloatTensor)
            styles = styles.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, styles_v = Variable(inputs.cuda(), requires_grad=False), Variable(styles.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, styles_v = Variable(inputs, requires_grad=False), Variable(styles, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, styles_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            if ite_num % print_freq == 0:
                print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0