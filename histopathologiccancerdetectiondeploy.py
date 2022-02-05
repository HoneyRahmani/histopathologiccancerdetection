# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:57:21 2020

@author: asus
"""
import histopathologiccancerdetection
import torch
import numpy as np
# Deploy the model for inference

# model parameters

params_model = {
        "input_shape":(3,96,96),
        "initial_filters": 8,
        "num_fc1" : 100,
        "dropout_rate": 0.25,
        "num_classes": 2,  
        }

cnn_model = histopathologiccancerdetection.Net(params_model)
path2weights ="./"
cnn_model.load_state_dict(torch.load(path2weights))
cnn_model.eval()

device = torch.device("cuda:0")
cnn_model=cnn_model.to(device)

import time
def deploy_model(model,dataset,device,num_classes=2,sanity_check=False):
    
    len_data = len(dataset)
    y_out = torch.zeros(len_data, num_classes)
    y_gt = np.zeros((len_data),dtype="uint8")
    model = model.to(device)
    
    elapsed_times=[]
    with torch.no_grad():
        
        for i in range(len_data):
            
            x,y=dataset[i]
            y_gt[i]=y
            start=time.time()
            y_out[i]=model(x.unsqueeze(0).to(device))
            elapsed = time.time()-start
            elapsed_times.append(elapsed)
            if sanity_check is True:
                break
    inference_time = np.mean(elapsed_times)*1000
    print("average inference time per image on %s:%.2f ms"
                  %(device,inference_time))
    return y_out.numpy(), y_gt

#
val_ds = histopathologiccancerdetection.val_ds
y_out, y_gt = deploy_model(cnn_model,val_ds,device=device,sanity_check=False)


from sklearn.metrics import accuracy_score

y_pred = np.argmax(y_out,axis=1)
acc = accuracy_score(y_pred,y_gt)


    
        
