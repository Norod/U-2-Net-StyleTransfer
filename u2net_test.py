import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import coremltools as ct

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import StylObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

FORCE_CPU = False
EXPORT_MODEL = False

def array_to_image(array_in):
    array_in = np.squeeze(255*(array_in))    
    array_in = np.transpose(array_in, (1, 2, 0))
    im = Image.fromarray(array_in.astype(np.uint8))
    return im

def save_output(image_name,pred,d_dir):

    predict_np = pred.cpu().data.numpy()
    im = array_to_image(predict_np)
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.jpg')

def main():

    # --------- 1. get image path and name ---------
    model_name='u2netp' #u2net
    #chkpnt_name = 'u2netp_bce_itr_8000_train_4.034824_tar_0.572258'
    #rescaleTransformVal = 512

    model_name='u2net' #u2netp
    chkpnt_name = 'u2net_bce_itr_16000_train_3.835149_tar_0.542587-400x_360x'
    rescaleTransformVal = 400
    

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, chkpnt_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_stylobj_dataset = StylObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(rescaleTransformVal),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_stylobj_dataloader = DataLoader(test_stylobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,3)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,3)

    if torch.cuda.is_available() and FORCE_CPU==False:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    
    net.eval()

    did_export = False
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_stylobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available() and FORCE_CPU==False:
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        if did_export == False and EXPORT_MODEL == True:
            traced_model = torch.jit.trace(net, inputs_test)
            m = torch.jit.script(traced_model)
            d1,d2,d3,d4,d5,d6,d7= m(inputs_test)
            jit_output = os.path.join(os.getcwd(), chkpnt_name + '.jit.pt') 
            torch.jit.save(m, jit_output)

            mlmodel_output = os.path.join(os.getcwd(), chkpnt_name + '.mlmodel') 
            model = ct.convert(m,inputs=[ct.TensorType(shape=inputs_test.shape)])
            model.save(mlmodel_output)

            onnx_output = os.path.join(os.getcwd(), chkpnt_name + '.onnx') 
            tensor_in = torch.Tensor(data_test['image'].type(torch.FloatTensor))
            print('shape=' + str(tensor_in.shape))
            torch.onnx.export(net, tensor_in, onnx_output, verbose=False, opset_version=11)

            did_export = True

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)

        prediction_dir_d1 = prediction_dir + 'd1' + os.sep
        if not os.path.exists(prediction_dir_d1):
            os.makedirs(prediction_dir_d1, exist_ok=True)

        prediction_dir_d2 = prediction_dir + 'd2' + os.sep
        if not os.path.exists(prediction_dir_d2):
            os.makedirs(prediction_dir_d2, exist_ok=True)

        # save results to test_results folder
        
        save_output(img_name_list[i_test],d1,prediction_dir_d1)
        save_output(img_name_list[i_test],d2,prediction_dir_d2)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
