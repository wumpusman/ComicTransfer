
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import deprecated
from torch.utils import data as data_torch
#
    

class TrainModel():
    #
    def __init__(self,model_to_learn,loss_function):
        """A simple wrapper for handling loss functions
        """
        self._loss_function=loss_function
        self.model=model_to_learn
    
    def set_loss_function(self,loss_function):
        """

        Args:
            loss_function:

        Returns:

        """
        self._loss_function=loss_function
    
    def prepare_loss(self,batch):
        """
        formats the data to be prepared in terms of the loss function shape
        Args:
            batch: a 2 by N Matrix where the first is the input, and the second is the output

        Returns:

        """
        x=batch[0]
        y=batch[1]
        x=x.permute((1,0,2,3,4))[0]
        y=y.permute((1,0,2,3,4))[0]

        predicted_y, features=self.model(x)

        return self._loss_function(predicted_y,y)
    
    
    def train(self,epochs,dataloader,optimizer,is_training=True)->[float]:
        """
        train a model or just run through a lose
        Args:
            epochs:
            dataloader:
            optimizer:
            is_training:

        Returns:

        """
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
            del loss #for memory management on my very limited GPU
            del x
            del y
            del z
            torch.cuda.empty_cache()
            loss_vals.append(np.array(temp_losses_batch).sum())
            print(loss_vals[-1])
        return loss_vals



class TrainWithFeatureChannels(TrainModel):

    def __init__(self,model_to_learn,loss_function):
        """
        trains a model but also expects an additional feature channel from earlier layer in the network
        it needs to be udpated
        Args:
            model_to_learn:  model that will be trained
            loss_function: loss function to be used
        """
        super().__init__(model_to_learn,loss_function)
            
    def prepare_loss(self,batch):
        x=batch[0]
        y=batch[1]
        z=batch[2]
        x=x.permute((1,0,2,3,4))[0]
        y=y.permute((1,0,2,3,4))[0]
        z=z.permute((1,0,2,3,4))[0] #earlier layer
       
        predicted_y, features=self.model(x)


        l1= self._loss_function(predicted_y,y)

        return l1
    
class TrainCoordWithUNET(TrainModel):
    def __init_(self,model_to_learn,loss_function):
        """
        trains model with craft text but also uses appended coordconv that has a slightly different output shape
        Args:
            model_to_learn:
            loss_function:

        Returns:

        """
        super().__init__(model_to_learn,loss_function)


    
    def prepare_loss(self,batch):
        x=batch[0]
        y=batch[1]
        x=x.permute((1,0,2,3,4))[0]
        y=y.permute((1,0,2,3,4))[0]

 
        predicted_y, features=self.model(x)
        predicted_y=predicted_y.permute((0,2,3,1)) #shape is different
  
        return self._loss_function(predicted_y,y)
    