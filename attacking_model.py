# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:17:52 2021

@author: asus
"""

# ==================== Load dataset
import mydataset

val_dl = mydataset.val_dl

for xb,yb in val_dl:
    print(xb.shape, yb.shape)
    break
#===================== Load the pre-trained model

import Mymodel

model = Mymodel.model


# ==================== Freez the model parametres

def freez_model(model):
    for child in model.children():
        for params in child.parameters():
            params.requires_grad = False
    print("Model frozen")
    return model

model = freez_model(model)

# ==================== Deploy the model on some data in dataset for test accuracy and loss before attacking
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def deploy_model (model, val_dl): 
    y_pred =[]
    y_gt = []
    
    with torch.no_grad():
        for x, y in val_dl:
            y_gt.append(y.item())
            out = model(x.to(device)).cpu().numpy()
            out = np.argmax(out, axis=1)[0]
            y_pred.append(out)
            
    return y_pred, y_gt
    
y_pred, y_gt = deploy_model(model, val_dl)

# ===================== Calculate accuracy

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_pred, y_gt)


# ===================== Implementing the attack

from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F
import matplotlib.pyplot as plt

y_pred = []
y_pred_p = []


def perturb_input(xb, yb, model, alfa):
    xb = xb.to(device)
    xb.requires_grad = True
    out = model(xb).cpu()
    loss = F.nll_loss(out, yb)
    model.zero_grad()
    loss.backward()
    xb_grad = xb.grad.data
    xb_p = xb + alfa * xb_grad.sign()
    xb_p = torch.clamp(xb_p, 0, 1)
    return xb_p, out.detach()

def attak_model(idx):
    i=0
    for xb, yb in val_dl:
        xb_p, out = perturb_input(xb, yb, model, alfa=0.005)
    
        i+=1
        with torch.no_grad():
    
            pred = out.argmax(dim=1, keepdim = False).item()
            y_pred.append(pred)
            prob = torch.exp(out[:,1])[0].item()
    
            out_p = model(xb_p).cpu()
            pred_p = out_p.argmax(dim=1, keepdim = False).item()
            y_pred_p.append(pred_p)
            prob_p = torch.exp(out_p[:,1])[0].item()
            
        # =============== Display orginal and pertrubed image
        if i == idx :
            xb_pp=xb
            xb_ppp = xb_p
            
            return xb_pp, xb_ppp, prob, prob_p

            
xb_pp, xb_ppp, prob, prob_p = attak_model(8)





# =============================================================================
plt.subplot(1,2,1)
plt.imshow(to_pil_image(xb_pp[0].detach().cpu()))
plt.title(prob)
plt.subplot(1,2,2)
plt.imshow(to_pil_image(xb_ppp[0].detach().cpu()))
plt.title(prob_p)
plt.show()
# =============================================================================

# ============= Calculate accuracy after image is pertrubed
from sklearn.metrics import accuracy_score

accuracy_p = accuracy_score(y_pred_p, y_pred)
print("accuracy before image is pertrubed", accuracy, "after", accuracy_p)

    


           
            
    
            
        



