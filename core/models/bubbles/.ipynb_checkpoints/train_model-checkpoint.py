
import torch
from torch.utils import data as data_torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


    

class TrainModel():
    
    def __init__(self,model_to_learn,loss_function):
        """A simple wrapper for handling loss functions
        """
        self._loss_function=loss_function
        self.model=model_to_learn
    
    def set_loss_function(self,loss_function):
        self._loss_function=loss_function
    
    def prepare_loss(self,batch):
        x=batch[0]
        y=batch[1]
        x=x.permute((1,0,2,3,4))[0]
        y=y.permute((1,0,2,3,4))[0]
        
        print(x.shape)
        print(y.shape)
        print("stop")
        predicted_y, features=self.model(x)
        print(predicted_y.shape)
        return self._loss_function(predicted_y,y)
    
    
    def train(self,epochs,dataloader,optimizer,is_training=True)->[float]:
        loss_vals=[]
        
        
        for epoch in range(epochs):
            cycle=(iter(dataloader))
            temp_losses_batch = []
            
            for i in range(len(cycle)):
                x,y,z=next(cycle)
                x=x.cuda()
                y=y.cuda()
                z=z.cuda()
                loss=None
                if is_training==True:
                    loss=self.prepare_loss((x,y,z)) #array so it can. be arbitrary features
                else:
                    with torch.no_grad():
                        loss=self.prepare_loss((x,y,z))
                
                temp_losses_batch.append(loss.cpu().detach().numpy())
                optimizer.zero_grad()

                if is_training==True:
                    loss.backward()
                    optimizer.step()
            del loss
            del x
            del y
            del z
            torch.cuda.empty_cache()
            loss_vals.append(np.array(temp_losses_batch).sum())
            print(loss_vals[-1])
        return loss_vals



class TrainWithFeatureChannels(TrainModel):
    
    def __init__(self,model_to_learn,loss_function):
            super().__init__(model_to_learn,loss_function)
            
    def prepare_loss(self,batch):
        x=batch[0]
        y=batch[1]
        z=batch[2]
        x=x.permute((1,0,2,3,4))[0]
        y=y.permute((1,0,2,3,4))[0]
        z=z.permute((1,0,2,3,4))[0]
       
        predicted_y, features=self.model(x)
        print("channel shape")
        
        #features=features.permute((1,0,2,3,4))[0]
        print(predicted_y.shape)
        print(y.shape)
        l1= self._loss_function(predicted_y,y)
        #l2= self._loss_function(features,z)
        return l1 #+(l2/32)
    
class TrainCoordWithUNET(TrainModel):
    def __init_(self,model_to_learn,loss_function):
        super().__init__(model_to_learn,loss_function)

    
    def prepare_loss(self,batch):
        x=batch[0]
        y=batch[1]
        x=x.permute((1,0,2,3,4))[0]
        y=y.permute((1,0,2,3,4))[0]

 
        predicted_y, features=self.model(x)
        predicted_y=predicted_y.permute((0,2,3,1))
  
        return self._loss_function(predicted_y,y)
    