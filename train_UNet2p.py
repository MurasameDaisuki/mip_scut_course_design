import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset,DataLoader
from model.UNet2p import UNetppL3
from model.Tracker import ReferringTracker
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
import cv2

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
img_size = 128
model = UNetppL3(h_dim=32).to(device)# torch.load('/mnt/disk2/SpineSagT2Wdataset3/weight/model_unetppl3.pth').to(device)#

weights_dict = torch.load("/mnt/disk2/SpineSagT2Wdataset3/weight/model_unetppl3.pth")#
model.load_state_dict(weights_dict, strict=False)

model_tracker = ReferringTracker(
        hidden_channel=1024,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
    ).to(device)
#weights_dict = torch.load('mip_course_design/model_tracker_02.pth')#
#model_tracker.load_state_dict(weights_dict, strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

optimizer_tracker = torch.optim.Adam(model_tracker.parameters(), lr=0.001)
scheduler_tracker = torch.optim.lr_scheduler.StepLR(optimizer_tracker, step_size=20, gamma=0.95)

def Dice_loss(target,predictive,ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss


def DSC(output,target,threshold=0.5):
    tp = np.sum(((output > threshold) == target) * target)
    fp = np.sum(((output > threshold) != target) * (1 - target))
    fn = np.sum(((output > threshold) != target) * target)
    return 2 * tp / (2 * tp + fp + fn)

def matrix(output,target,threshold=0.6):
    tp = np.sum(((output > threshold) == target) * target)
    tn = np.sum(((output > threshold) == target) * (1 - target))
    fp = np.sum(((output > threshold) != target) * (1 - target))
    fn = np.sum(((output > threshold) != target) * target)
    return tp,tn,fp,fn


class TrainSet(Dataset):
    def __init__(self, X, Y):
        # 定义好 image 的路径
        self.X, self.Y = X, Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

def train(train_list,model,img_size=img_size,set_size=16,batch_size=2):
    X_train = []
    y_train = []

    for subject in tqdm(train_list):
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'/mnt/disk2/SpineSagT2Wdataset3/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()
 
        X_train.append(img)
        y_train.append(target)

    X_train_resize = []
    y_train_resize = []

    for i in range(len(train_list)):
        X_train_resize.append([])
        y_train_resize.append([])
        for j in range(X_train[i].shape[-1]):
            X_train_resize[-1].append([cv2.resize(X_train[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            y_train_resize[-1].append([cv2.resize(y_train[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            
            if np.random.random(1)[0] <= 0.3:
                X_train_resize[-1].append([cv2.resize(X_train[i][:,::-1,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
                y_train_resize[-1].append([cv2.resize(y_train[i][:,::-1,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            if np.random.random(1)[0] <= 0.3:
                X_train_resize[-1].append([cv2.resize(X_train[i][::-1,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
                y_train_resize[-1].append([cv2.resize(y_train[i][::-1,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            if np.random.random(1)[0] <= 0.3:
                X_train_resize[-1].append([cv2.resize(X_train[i][::-1,::-1,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
                y_train_resize[-1].append([cv2.resize(y_train[i][::-1,::-1,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            
    del X_train
    X_train_resize = np.array(X_train_resize)
    y_train_resize = np.array(y_train_resize)
    resizeimg = np.concatenate(X_train_resize)
    #print(resizeimg.shape)
    resizetarget = np.concatenate(y_train_resize)

    np.random.seed(13)
    np.random.shuffle(resizeimg)
    np.random.seed(13)
    np.random.shuffle(resizetarget)

    resizeimg = torch.tensor(resizeimg)
    resizetarget = torch.tensor(resizetarget)

    trainset = TrainSet(resizeimg, resizetarget)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    meanloss0 = 0
    meanloss1 = 0
    meanloss2 = 0
   
    dsc_list = []
    add_num = 0

    for data,target in tqdm(train_loader):
        #print(data.shape,target.shape)
        #optimizer.zero_grad()
        model.train()

        data = data.type(torch.FloatTensor).to(device)
        target = target.type(torch.FloatTensor).to(device)

        output = model(data)

        loss_fn = nn.MSELoss()

        loss0 = loss_fn(output[0], target)# + Dice_loss(output, target)
        loss0.backward(retain_graph=True)

        loss1 = loss_fn(output[1], target)# + Dice_loss(output, target)
        loss1.backward(retain_graph=True)

        loss2 = loss_fn(output[2], target)# + Dice_loss(output, target)
        loss2.backward()

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        #add_num = 0

        meanloss0 += loss0.item() / (len(train_loader))
        meanloss1 += loss1.item() / (len(train_loader))
        meanloss2 += loss2.item() / (len(train_loader))

        target = target.cpu().detach().numpy()
        output = output[2].cpu().detach().numpy()
        dsc = DSC(output, target, threshold=0.6)
        dsc_list.append(dsc)

    scheduler.step()
    return meanloss0,meanloss1,meanloss2,np.mean(dsc_list)

def train_tracker(X_train_resize,y_train_resize,model_seg,model_tracker,img_size=img_size):

    model_seg.eval()
    model_tracker.train()

    meanloss = 0

    dsc_3_list = []
    dsc_4_list = []
    dsc_5_list = []
    dsc_6_list = []
    dsc_7_list = []
    dsc_8_list = []
    dsc_9_list = []
    for subject in tqdm(range(len(X_train_resize))):
        data = torch.tensor(X_train_resize[subject]).to(device).type(torch.FloatTensor).to(device)
        target = torch.tensor(y_train_resize[subject]).to(device).type(torch.FloatTensor).to(device)
        #print(data.shape,target.shape)
        output_seg = None
        with torch.no_grad():
            output_seg = model_seg(data)[-1]
        
        output = model_tracker(output_seg)

        loss_fn = nn.MSELoss()

        loss = loss_fn(output, target)# + Dice_loss(output, target)
        loss.backward(retain_graph=True)

        optimizer_tracker.step()
        optimizer_tracker.zero_grad()
        torch.cuda.empty_cache()
        #add_num = 0

        meanloss += loss.item() / (len(X_train_resize))

        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()


        dsc = DSC(output,target,threshold=0.3)
        dsc_3_list.append(dsc)
        dsc = DSC(output,target,threshold=0.4)
        dsc_4_list.append(dsc)
        dsc = DSC(output,target,threshold=0.5)
        dsc_5_list.append(dsc)
        dsc = DSC(output,target,threshold=0.6)
        dsc_6_list.append(dsc)
        dsc = DSC(output,target,threshold=0.7)
        dsc_7_list.append(dsc)
        dsc = DSC(output,target,threshold=0.1)
        dsc_8_list.append(dsc)
        dsc = DSC(output,target,threshold=0.2)
        dsc_9_list.append(dsc)


    scheduler_tracker.step()
    result = (
        round(np.mean(dsc_3_list),ndigits=6),
        round(np.mean(dsc_4_list),ndigits=6),
        round(np.mean(dsc_5_list),ndigits=6),
        round(np.mean(dsc_6_list),ndigits=6),
        round(np.mean(dsc_7_list),ndigits=6),
        round(np.mean(dsc_8_list),ndigits=6),
        round(np.mean(dsc_9_list),ndigits=6),
    )
    return meanloss, result


def test_tracker(X_test_resize,y_test_resize,model_seg,model_tracker,img_size=img_size):

    model_seg.eval()
    model_tracker.eval()

    meanloss = 0
    dsc_1_list = []
    dsc_2_list = []
    dsc_3_list = []
    dsc_4_list = []
    dsc_5_list = []
    dsc_6_list = []
    dsc_7_list = []

    for subject in tqdm(range(len(X_test_resize))):
        data = torch.tensor(X_test_resize[subject]).to(device).type(torch.FloatTensor).to(device)
        target = torch.tensor(y_test_resize[subject]).to(device).type(torch.FloatTensor).to(device)
        #print(data.shape,target.shape)
        output_seg = None
        output = None
        with torch.no_grad():
            output_seg = model_seg(data)[-1]
            output = model_tracker(output_seg)
        torch.cuda.empty_cache()
        #add_num = 0

        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        dsc = DSC(output,target,threshold=0.3)
        dsc_3_list.append(dsc)
        dsc = DSC(output,target,threshold=0.4)
        dsc_4_list.append(dsc)
        dsc = DSC(output,target,threshold=0.5)
        dsc_5_list.append(dsc)
        dsc = DSC(output,target,threshold=0.6)
        dsc_6_list.append(dsc)
        dsc = DSC(output,target,threshold=0.7)
        dsc_7_list.append(dsc)
        dsc = DSC(output,target,threshold=0.1)
        dsc_1_list.append(dsc)
        dsc = DSC(output,target,threshold=0.2)
        dsc_2_list.append(dsc)



    result = (
        round(np.mean(dsc_1_list),ndigits=6),
        round(np.mean(dsc_2_list),ndigits=6),
        round(np.mean(dsc_3_list),ndigits=6),
        round(np.mean(dsc_4_list),ndigits=6),
        round(np.mean(dsc_5_list),ndigits=6),
        round(np.mean(dsc_6_list),ndigits=6),
        round(np.mean(dsc_7_list),ndigits=6),

    )
    return result


def test(test_list,model,img_size=img_size,set_size=16,batch_size=2):
    X_test = []
    y_test = []

    for subject in tqdm(test_list):
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'/mnt/disk2/SpineSagT2Wdataset3/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()

        X_test.append(img)
        y_test.append(target)

    X_test_resize = []
    y_test_resize = []

    for i in range(len(test_list)):
        X_test_resize.append([])
        y_test_resize.append([])
        for j in range(X_test[i].shape[-1]):
            X_test_resize[-1].append([cv2.resize(X_test[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            y_test_resize[-1].append([cv2.resize(y_test[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])

    del X_test
    X_test_resize = np.array(X_test_resize)
    y_test_resize = np.array(y_test_resize)

    resizeimg = np.concatenate(X_test_resize)
    resizetarget = np.concatenate(y_test_resize)

    np.random.seed(13)
    np.random.shuffle(resizeimg)
    np.random.seed(13)
    np.random.shuffle(resizetarget)

    resizeimg = torch.tensor(resizeimg)
    resizetarget = torch.tensor(resizetarget)

    testset = TrainSet(resizeimg, resizetarget)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    dsc_3_list = []
    dsc_4_list = []
    dsc_5_list = []
    dsc_6_list = []
    dsc_7_list = []
    dsc_8_list = []
    dsc_9_list = []


    for data,target in test_loader:
        #print(data.shape,target.shape)
        with torch.no_grad():
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            model.eval()

            data = data.type(torch.FloatTensor).to(device)
            target = target.type(torch.FloatTensor).to(device)


            output = model(data)[2]
            target = target.cpu().numpy()
            output = output.cpu().numpy()
            dsc = DSC(output,target,threshold=0.3)
            dsc_3_list.append(dsc)
            dsc = DSC(output,target,threshold=0.4)
            dsc_4_list.append(dsc)
            dsc = DSC(output,target,threshold=0.5)
            dsc_5_list.append(dsc)
            dsc = DSC(output,target,threshold=0.6)
            dsc_6_list.append(dsc)
            dsc = DSC(output,target,threshold=0.7)
            dsc_7_list.append(dsc)
            dsc = DSC(output,target,threshold=0.8)
            dsc_8_list.append(dsc)
            dsc = DSC(output,target,threshold=0.9)
            dsc_9_list.append(dsc)
    result = (
        round(np.mean(dsc_3_list),ndigits=6),
        round(np.mean(dsc_4_list),ndigits=6),
        round(np.mean(dsc_5_list),ndigits=6),
        round(np.mean(dsc_6_list),ndigits=6),
        round(np.mean(dsc_7_list),ndigits=6),
        round(np.mean(dsc_8_list),ndigits=6),
        round(np.mean(dsc_9_list),ndigits=6),
    )
    return result, matrix(output,target,threshold=0.6)



def predit(img_path,model,img_size=img_size):

    #img_path = r'/mnt/disk2/SpineSagT2Wdataset3/test_process/Case' + str(subject) + '.nii.gz'

    img = nib.load(img_path)
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
    predit_pic = predit_pic > 0.65



if __name__ == '__main__':

    train_list = range(1, 130)
    test_list = range(130, 153)
    vaild_list = range(153, 175)

    X_train = []
    y_train = []

    for subject in tqdm(train_list):
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'/mnt/disk2/SpineSagT2Wdataset3/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()
 
        X_train.append(img)
        y_train.append(target)

    X_train_resize = []
    y_train_resize = []

    for i in range(len(train_list)):
        X_train_subject = []
        y_train_subject = []
        for j in range(X_train[i].shape[-1]):
            X_train_subject.append([cv2.resize(X_train[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            y_train_subject.append([cv2.resize(y_train[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
        X_train_resize.append(X_train_subject)
        y_train_resize.append(y_train_subject)

    
    X_test = []
    y_test = []
    
    for subject in tqdm(test_list):
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'/mnt/disk2/SpineSagT2Wdataset3/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()
 
        X_test.append(img)
        y_test.append(target)

    X_test_resize = []
    y_test_resize = []

    for i in range(len(test_list)):
        X_test_subject = []
        y_test_subject = []
        for j in range(X_test[i].shape[-1]):
            X_test_subject.append([cv2.resize(X_test[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
            y_test_subject.append([cv2.resize(y_test[i][:,:,j], (256, 256), interpolation=cv2.INTER_NEAREST)])
        X_test_resize.append(X_test_subject)
        y_test_resize.append(y_test_subject)

    for i in range(2000):
        trainloss0, trainloss1, trainloss2, trainDSC = train(train_list, model, set_size=8, batch_size=16)
        print("Epochs:", i, "trainloss:", trainloss0, trainloss1, trainloss2, "trainDSC:", trainDSC)
        model.eval()
        testDSC = test(test_list, model, set_size=8, batch_size=8)
        print("Epochs:", i, "test_wholepicDSC:", testDSC[0])
        if i % 5 == 0:
            torch.save(model.state_dict(), '/mnt/disk2/SpineSagT2Wdataset3/weight/model_unetppl3_01.pth')
            testDSC = test(vaild_list, model, set_size=8, batch_size=8)
            print("Epochs:", i, "test_wholepicDSC:", testDSC[0])
            print("Epochs:", i, f"test_wholepicMatrix tp:{testDSC[1][0]},tn:{testDSC[1][1]},fp:{testDSC[1][2]},fn:{testDSC[1][3]}")
