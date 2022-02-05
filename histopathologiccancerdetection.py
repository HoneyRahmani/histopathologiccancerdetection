# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:44:57 2020

@author: asus
"""

# Import important module

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import os


# Initialize

path2csv = "./data/train_labels.csv"
labels_df = pd.read_csv(path2csv)
labels_df.head()

print(labels_df['label'].value_counts())
labels_df['label'].hist()

malignantIds = labels_df.loc[labels_df['label']==1]['id'].values
path2train= "./data/train/"
color = False

plt.rcParams['figure.figsize'] = (10.0,10.0)
plt.subplots_adjust(wspace=0, hspace=0)
nrows,ncols = 3,3

#the center 32 x 32 region of an image contains at least one pixel of tumor tissue.
for i,id_ in enumerate(malignantIds[:nrows*ncols]):
    
    full_filenames = os.path.join(path2train, id_+'.tif')
    
    img = Image.open(full_filenames)
    
    draw = ImageDraw.Draw(img)
    draw.rectangle(((32,32),(64,64)), outline="green")
    plt.subplot(nrows, ncols, i+1)
    
    if color is True:
        plt.imshow(np.array(img))
    else:
        plt.imshow(np.array(img)[:,:,0], cmap="gray")
    plt.axis('off')

print("image shape:", np.array(img).shape)
print("pixel values range from %s to %s" %(np.min(img),np.max(img)))
#=======================================================================================    
# Custom Dataset class
# import important module

from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os 


torch.manual_seed(0)

class histoCancerDataset(Dataset):
    
    def __init__(self, data_dire, transform, data_type="train"):
        
        path2data = os.path.join(data_dire,data_type)
        filenames = os.listdir(path2data)
        self.full_filenames = [os.path.join(path2data,f) for f in filenames]
        
        csv_filename = data_type + "_labels.csv"
        path2csvLabels = os.path.join(data_dire,csv_filename)
        labels_df = pd.read_csv(path2csvLabels)
        
        labels_df.set_index("id", inplace=True)
        
        self.labels = [labels_df.loc[filenames[:-4]].values[0] for 
                       filenames in filenames]
        
        self.transform = transform
        
    def __len__(self):
            
            return len(self.full_filenames)
        
    def __getitem__(self, idx):
            
            image = Image.open(self.full_filenames[idx])
            image = self.transform(image)
            return image, self.labels[idx]
        
# Initialize for calling above class
    
data_transformer = transforms.Compose([transforms.ToTensor()])
data_dir = "./data/"
histo_dataset = histoCancerDataset(data_dir, data_transformer, "train")
print(len(histo_dataset))

img,label = histo_dataset[9]
print(img.shape, torch.min(img), torch.max(img)) 

#=================================================================
# Splitting the dataset to train and validation

from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

np.random.seed(0) 
len_histo = len(histo_dataset)
len_train = int(0.8 * len_histo)
len_val = len_histo - len_train
#"random_split" randomly split a dataset into non-overlapping new datasets of given lengths
train_ds, val_ds = random_split(histo_dataset, [len_train,len_val])
# Show image function

def show(img, y, color = False):
    
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    if color == False:
        npimg_tr = npimg_tr[:,:,0]
        plt.imshow(npimg_tr, interpolation='nearest', cmap ='gray')
    else:
        plt.imshow(npimg_tr, interpolation='nearest')
    plt.title("label: " + str(y))
#show train image   
grid_size = 4
rnd_inds = np.random.randint(0,len(train_ds),grid_size)
x_grid_train = [train_ds[i][0] for i in rnd_inds]
y_grid_train = [train_ds[i][1] for i in rnd_inds]
    
x_grid_train = utils.make_grid(x_grid_train, nrow=4, padding=2)
plt.rcParams['figure.figsize'] = (10.0, 5)
show(x_grid_train, y_grid_train)

#show validation image
grid_size_val = 4
rnd_inds_val = np.random.randint(0,len(val_ds),grid_size_val)
x_grid_val = [val_ds[i][0] for i in range(grid_size_val)]
y_grid_val = [val_ds[i][1] for i in range(grid_size_val)]
    
x_grid_val = utils.make_grid(x_grid_val, nrow=4, padding=2)
show(x_grid_val, y_grid_val)

# =========================================================
#Augmentation

train_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(96,scale=(0.8,1.0),ratio=(1.0,1.0)),
        transforms.ToTensor()])
val_transformer = transforms.Compose([transforms.ToTensor()])

train_ds.transform = train_transformer
val_ds.transform = val_transformer
#============================================================
#Creating dataloaders

from torch.utils.data import DataLoader
train_dl = DataLoader(train_ds, batch_size=32,shuffle=True)
val_dl = DataLoader(val_ds,batch_size=64,shuffle=False)

#==============================================================
#calculate the output size of a CNN layer
import torch.nn as nn
import numpy as np
def findConv2dOutShape(H_in,W_in,conv,pool=2):
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation
    #hight of output image
    H_out = np.floor((H_in+2*padding[0]-
                      dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    #width of output image
    W_out = np.floor((W_in+2*padding[1]-
                      dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
    
    if pool:
        H_out/=pool
        W_out/=pool
    return int(H_out),int(W_out)

#Building the classification model
#import torch.nn as nn
import torch.nn.functional as F
 
class Net(nn.Module):
     
    def __init__(self, params):
         
         super(Net, self).__init__()
         C_in,H_in,W_in = params["input_shape"]
         init_f = params["initial_filters"]
         num_fc1 = params["num_fc1"]
         num_classes = params["num_classes"]
         self.droupout_rate = params["dropout_rate"]
         
         self.conv1 = nn.Conv2d(C_in,init_f,kernel_size=3)
         h,w=findConv2dOutShape(H_in,W_in,self.conv1)
         self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
         h,w=findConv2dOutShape(h,w,self.conv2)
         self.conv3 = nn.Conv2d(2*init_f,4*init_f, kernel_size=3)
         h,w=findConv2dOutShape(h,w,self.conv3)
         self.conv4 = nn.Conv2d(4*init_f, 8*init_f,kernel_size=3)
         h,w=findConv2dOutShape(h,w,self.conv4)
         #use the findConv2DOutShape function for compute num_flatten
         self.num_flatten = h*w*8*init_f
         self.fc1 = nn.Linear(self.num_flatten, num_fc1)
         self.fc2 = nn.Linear(num_fc1, num_classes)
         
    def forward(self, x):
            
         x = F.relu(self.conv1(x))
         x = F.max_pool2d(x,2,2)
         x = F.relu(self.conv2(x))
         x = F.max_pool2d(x,2,2)
         x = F.relu(self.conv3(x))
         x = F.max_pool2d(x,2,2)
         x = F.relu(self.conv4(x))
         x = F.max_pool2d(x,2,2)
         x = x.view(-1, self.num_flatten)
         x = F.relu(self.fc1(x))
         x = F.dropout(x, self.droupout_rate, training= self.training)
         x = self.fc2(x)
         return F.log_softmax(x,dim=1)
     
# construct an object
params_model = {
    "input_shape": (3,96,96),
    "initial_filters":8,
    "num_fc1": 100,
    "dropout_rate":0.25,
    "num_classes":2,                
                }
'''import torch 
x = torch.randn(128, 20)

x = x.view(-1, 2560)
print(x.shape)'''
device=torch.device("cuda:0")
cnn_model = Net(params_model)
cnn_model = cnn_model.to(device)
#print model is useful for verify __init__ function in Net class
print (cnn_model)
print("Model's state_dict:")
for param_tensor in cnn_model.state_dict():
    print(param_tensor, "\t", cnn_model.state_dict()[param_tensor].size())



#summary model is useful also for verify forward function in Net class            
from torchsummary import summary
summary(cnn_model, input_size=(3,96,96),device=device.type)

#==============================================================
# Train and evalution

def metrics_batch(output, target):
    
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func,output, target, opt=None):
    
    loss = loss_func(output,target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None: 
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

def loss_epoch(model, loss_func, dataset_dl, sanity_check = False, opt=None):
    
    running_loss = 0.0
    running_metric =0.0
    len_data =len(dataset_dl.dataset)
    
    for xb,yb in dataset_dl:
        
        xb = xb.to(device)
        yb = yb.to(device)
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func,output,yb,opt)
        
        running_loss+=loss_b
        if metric_b is not None:
            running_metric+=metric_b
        
        if sanity_check is True:
            break
        
    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    
    return loss,metric
def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']
    
def train_val(model, params):

        num_epochs = params["num_epochs"]
        loss_func = params["loss_func"]
        opt = params["optimizer"]
        train_dl = params["train_dl"]
        val_dl=params["val_dl"]
        sanity_check = params["sanity_check"]
        lr_scheduler = params["lr_scheduler"]
        path2weights = params["path2weights"]
        
        loss_history = {
                "train": [],
                "val": [],
                }
        
        metric_history = {
                "train": [],
                "val": [],
                } 
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        

        for epoch in range(num_epochs):
            
            current_lr = get_lr(opt)
            
            print('Epoch {}/{}, curren lr = {}|'.format(
                   epoch,num_epochs-1,current_lr))
            
            model.train()
            train_loss, train_metric = loss_epoch(model,loss_func,train_dl,sanity_check,opt)
            
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)
            
            model.eval()
            with torch.no_grad():
                val_loss, val_metric = loss_epoch(model,loss_func,val_dl,sanity_check)
                loss_history["val"].append(val_loss)
                metric_history["val"].append(val_metric)
                
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),path2weights)
                print("Copied best model weights!")
            
            
            lr_scheduler.step(val_loss)
            if current_lr != get_lr(opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)
            print("train loss: %.6f, validation loss: %.f, accuracy:%.2f"%
               (train_loss, val_loss, 100*val_metric))
            print("-"*10)
        model.load_state_dict(best_model_wts)
        return model,loss_history,metric_history

#===================================================================
import copy
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(),lr = 3e-4)

print("Optimizer's state_dict:")
for var_name in opt.state_dict():
    print(var_name, "\t", opt.state_dict()[var_name])


lr_scheduler = ReduceLROnPlateau(opt, mode = 'min', factor=0.5,
                                 patience=20,verbose=1)

params_train = {
        "num_epochs":100,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "sanity_check": False,
        "lr_scheduler": lr_scheduler,
        "path2weights": "./weights.pt",
        }

cnn_model,loss_hist,metric_hist = train_val(cnn_model,params_train)

num_epochs = params_train["num_epochs"]

#plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1), loss_hist["train"], label = "train")
plt.plot(range(1,num_epochs+1), loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"], label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"], label = "val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.grid()
plt.show()
                        
                    
#=======================================================

# =============================================================================
# loss_func = nn.NLLLoss(reduction="sum")
# opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
# lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)
# 
# params_train={
#  "num_epochs": 2,
#  "optimizer": opt,
#  "loss_func": loss_func,
#  "train_dl": train_dl,
#  "val_dl": val_dl,
#  "sanity_check": False,
#  "lr_scheduler": lr_scheduler,
#  "path2weights": "./models/weights.pt",
# }
# 
# # train and validate the model
# cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train)
# 
# # Train-Validation Progress
# num_epochs=params_train["num_epochs"]

# 
# # plot loss progress
# plt.title("Train-Val Loss")
# plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
# plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
# plt.ylabel("Loss")
# plt.xlabel("Training Epochs")
# plt.legend()
# plt.show()
# 
# # plot accuracy progress
# plt.title("Train-Val Accuracy")
# plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
# plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
# plt.ylabel("Accuracy")
# plt.xlabel("Training Epochs")
# plt.legend()
# plt.show()
# =============================================================================



        






    
    
    
                            
    


  
            
        
        
        


