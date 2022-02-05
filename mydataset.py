# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:08:30 2021

@author: asus
"""

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
#=================================================================
# Splitting the dataset to train and validation

from torch.utils.data import random_split
import numpy as np
np.random.seed(0) 
len_histo = len(histo_dataset)
len_train = int(0.8 * len_histo)
len_val = len_histo - len_train
train_ds, val_ds = random_split(histo_dataset, [len_train,len_val])

#===========================================================
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
val_dl = DataLoader(val_ds,batch_size=1,shuffle=False)