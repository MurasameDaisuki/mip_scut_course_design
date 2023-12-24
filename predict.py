import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset,DataLoader
from model.ACC_UXNet import ACC_UXNet
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
import SimpleITK as sitk

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
img_size = 128#/opt/anaconda3/envs/torch/bin/python /home/zyb/mip_course_design/ablation.py/opt/anaconda3/envs/torch/bin/python /home/zyb/mip_course_design/main.py
model = ACC_UXNet(h_dim=64,input_size=(img_size,img_size,12)).to(device)# torch.load('model.pth').to(device)#

weights_dict = torch.load('/mnt/disk2/SpineSagT2Wdataset3/weight/model_cross_01.pth')#
model.load_state_dict(weights_dict, strict=False)


def predit(img_path,model,output_path,img_size=img_size):

    #img_path = r'/mnt/disk2/SpineSagT2Wdataset3/test_process/Case' + str(subject) + '.nii.gz'

    img = nib.load(img_path)
    img_affine  = img.affine

    img = img.get_fdata()
    img /= np.max(img)

    h, w, l = img.shape
    predit_pic = np.zeros_like(img)
    num_compute = np.zeros_like(img)

    overlap = int((h-128)/8)

    with torch.no_grad():
        for h in range(9):
            test_batch = []
            for w in range(9):
                test_batch.append([img[h*overlap:h*overlap+img_size,w*overlap:w*overlap+img_size,:12]])
            
            model.eval() 
            test_batch = torch.tensor(test_batch).to(device)
            test_batch = test_batch.type(torch.FloatTensor).to(device)
            output = model(test_batch)
            output = output.cpu().numpy()
            for w in range(9):
                num_compute[h*overlap:h*overlap+img_size,w*overlap:w*overlap+img_size,:12] += np.ones((img_size,img_size,12))
                predit_pic[h*overlap:h*overlap+img_size,w*overlap:w*overlap+img_size,:12] += output[w][0]
        
        if l != 12:
            for h in range(9):
                test_batch = []
                for w in range(9):
                    test_batch.append([img[h*overlap:h*overlap+img_size,w*overlap:w*overlap+img_size,-12:]])
                
                model.eval() 
                test_batch = torch.tensor(test_batch).to(device)
                test_batch = test_batch.type(torch.FloatTensor).to(device)
                output = model(test_batch)
                output = output.cpu().numpy()
                for w in range(9):
                    num_compute[h*overlap:h*overlap+img_size,w*overlap:w*overlap+img_size,-12:] += np.ones((img_size,img_size,12))
                    predit_pic[h*overlap:h*overlap+img_size,w*overlap:w*overlap+img_size,-12:] += output[w][0]

    predit_pic = predit_pic/num_compute
    predit_pic[predit_pic>0.6]=1
    predit_pic[predit_pic<0.6]=0  
    predit_pic = predit_pic.astype(np.float64)
    print(predit_pic.shape)

    #predit_pic = sitk.GetImageFromArray(predit_pic)
    #sitk.WriteImage(predit_pic, output_path)
    nib.Nifti1Image(predit_pic,img_affine).to_filename(output_path)

    print(output_path)

if __name__ == '__main__':
    for subject in range(176,196):
        print(subject)
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/test_process/Case' + str(subject) + '.nii.gz'
        output_path = r'/mnt/disk2/SpineSagT2Wdataset3/test_predict/mask_case' + str(subject) + '.nii.gz'
        predit(img_path,model,output_path,img_size=img_size)