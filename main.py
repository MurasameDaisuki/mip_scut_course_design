import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset,DataLoader
from model.ACC_UXNet import ACC_UXNet
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
img_size = 128
model = ACC_UXNet(h_dim=64,input_size=(img_size,img_size,12)).to(device)# torch.load('model.pth').to(device)#

#weights_dict = torch.load('weight/model_cross_01.pth')#
#model.load_state_dict(weights_dict, strict=False)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)


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

    for subject in train_list:
        img_path = r'dataset/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'dataset/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()

        X_train.append(img)
        y_train.append(target)

    newimg = []
    newtarget = []

    while len(newimg) < set_size*len(train_list):
        for i in range(len(X_train)):
            img = X_train[i]
            target = y_train[i]
            h_min = np.random.randint(0, img.shape[0] - img_size)
            w_min = np.random.randint(0, img.shape[1] - img_size)
            if np.sum(img[h_min:h_min + img_size, w_min:w_min + img_size, :12] > 0) / (img_size * img_size * 12) >= 0.7:
                if np.sum(target[h_min:h_min + img_size, w_min:w_min + img_size, :12] > 0) / (img_size * img_size * 12) >= 0.01:
                    newimg_patch = img[h_min:h_min + img_size, w_min:w_min + img_size, :12]
                    newtarget_patch = target[h_min:h_min + img_size, w_min:w_min + img_size, :12]
                    if np.random.random(1)[0] <= 0.3:
                        newimg_patch = newimg_patch[:,:,::-1]
                        newtarget_patch = newtarget_patch[:,:,::-1]
                    if np.random.random(1)[0] <= 0.3:
                        newimg_patch = newimg_patch[:,::-1,:]
                        newtarget_patch = newtarget_patch[:,::-1,:]
                    if np.random.random(1)[0] <= 0.3:
                        newimg_patch = newimg_patch[::-1,:,:]
                        newtarget_patch = newtarget_patch[::-1,:,:]
                    newimg.append([newimg_patch])
                    newtarget.append([newtarget_patch])
                    #



                else:
                    pass
            else:
                pass

    newimg = np.array(newimg)
    newtarget = np.array(newtarget)

    np.random.seed(114514) 
    np.random.shuffle(newimg)
    np.random.seed(114514) 
    np.random.shuffle(newtarget)

    newimg = torch.tensor(newimg)
    newtarget = torch.tensor(newtarget)


    trainset = TrainSet(newimg, newtarget)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    meanloss = 0
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
        loss = loss_fn(output, target)# + Dice_loss(output, target)

        loss.backward()
        #add_num += 1
        #if add_num >= 16:
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        #add_num = 0

        meanloss += loss.item() / (len(train_loader))

        target = target.cpu().detach().numpy()
        output = output.cpu().detach().numpy()
        dsc = DSC(output, target, threshold=0.3)
        dsc_list.append(dsc)

    scheduler.step()
    return meanloss,np.mean(dsc_list)




def test(test_list,model,img_size=img_size,set_size=16,batch_size=2):
    X_test = []
    y_test = []

    for subject in test_list:
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'/mnt/disk2/SpineSagT2Wdataset3/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()

        X_test.append(img)
        y_test.append(target)

    newimg = []
    newtarget = []

    while len(newimg) < set_size*len(test_list):
        for i in range(len(X_test)):
            img = X_test[i]
            target = y_test[i]
            h_min = np.random.randint(0, img.shape[0] - img_size)
            w_min = np.random.randint(0, img.shape[1] - img_size)
            if np.sum(img[h_min:h_min + img_size, w_min:w_min + img_size, :12] > 0) / (img_size * img_size * 12) >= 0.7:
                if np.sum(target[h_min:h_min + img_size, w_min:w_min + img_size, :12] > 0) / (img_size * img_size * 12) >= 0.01:
                    newimg.append([img[h_min:h_min + img_size, w_min:w_min + img_size, :12]])
                    newtarget.append([target[h_min:h_min + img_size, w_min:w_min + img_size, :12]])
                else:
                    pass
            else:
                pass
    newimg = torch.tensor(newimg)
    newtarget = torch.tensor(newtarget)

    testset = TrainSet(newimg, newtarget)
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


            output = model(data)
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
    return result



def test_wholepic(test_list,model,img_size=img_size,set_size=16,batch_size=2):
    X_test = []
    y_test = []

    for subject in test_list:
        img_path = r'/mnt/disk2/SpineSagT2Wdataset3/image_process/Case' + str(subject) + '.nii.gz'
        target_path = r'/mnt/disk2/SpineSagT2Wdataset3/groundtruth/mask_case' + str(subject) + '.nii.gz'

        img = nib.load(img_path)
        img = img.get_fdata()
        img /= np.max(img)

        target = nib.load(target_path)
        target = target.get_fdata()

        X_test.append(img)
        y_test.append(target)


    dsc_3_list = []
    dsc_4_list = []
    dsc_5_list = []
    dsc_6_list = []
    dsc_7_list = []
    dsc_8_list = []
    dsc_9_list = []


    for i in range(len(X_test)):
        #print(X_test[i].shape)
        img = X_test[i][:880,:880,:12]
        #print(img.shape)
        target = y_test[i][:880,:880,:12]
        predit_pic = np.zeros_like(img)
        num_compute = np.zeros_like(img)

        with torch.no_grad():
            for h in range(9):
                test_batch = []
                for w in range(9):
                    test_batch.append([img[h*94:h*94+img_size,w*94:w*94+img_size,:12]])
                
                model.eval() 
                test_batch = torch.tensor(test_batch).to(device)
                test_batch = test_batch.type(torch.FloatTensor).to(device)
                output = model(test_batch)
                output = output.cpu().numpy()
                for w in range(9):
                    num_compute[h*94:h*94+img_size,w*94:w*94+img_size,:12] += np.ones((img_size,img_size,12))
                    predit_pic[h*94:h*94+img_size,w*94:w*94+img_size,:12] += output[w][0]

        predit_pic = predit_pic/num_compute
        
        dsc = DSC(predit_pic,target,threshold=0.3)
        dsc_3_list.append(dsc)
        dsc = DSC(predit_pic,target,threshold=0.4)
        dsc_4_list.append(dsc)
        dsc = DSC(predit_pic,target,threshold=0.5)
        dsc_5_list.append(dsc)
        dsc = DSC(predit_pic,target,threshold=0.6)
        dsc_6_list.append(dsc)
        dsc = DSC(predit_pic,target,threshold=0.7)
        dsc_7_list.append(dsc)
        dsc = DSC(predit_pic,target,threshold=0.8)
        dsc_8_list.append(dsc)
        dsc = DSC(predit_pic,target,threshold=0.9)
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
    return result, matrix(predit_pic,target,threshold=0.6)




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
    for i in range(2000):

        trainloss, trainDSC = train(random.sample(range(1, 130),64), model, set_size=16, batch_size=8)
        print("Epochs:", i, "trainloss:", trainloss, "trainDSC:", trainDSC)
        model.eval()
        torch.save(model.state_dict(), 'weight/model_cross_01.pth')
        #cross_01: 1, 141
        if i % 3 == 0:
            testDSC = test(random.sample(range(130, 153),8), model, set_size=8, batch_size=8)
            print("Epochs:", i, "testDSC:", testDSC)

        if i % 10 == 0:
            testDSC = test_wholepic(range(153, 175), model, set_size=16, batch_size=8)
            print("Epochs:", i, "test_wholepicDSC:", testDSC[0])
            print("Epochs:", i, f"test_wholepicMatrix tp:{testDSC[1][0]},tn:{testDSC[1][1]},fp:{testDSC[1][2]},fn:{testDSC[1][3]}")
        