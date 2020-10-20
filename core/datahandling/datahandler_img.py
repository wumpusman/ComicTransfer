import sys
import pandas as pd
import numpy as np
import sklearn as sk
import torch
import os
from os import listdir
from os.path import isfile, join
import torch
from torch.utils import data as data_torch
from craft_text_detector import image_utils as imgproc
import torch
from torch.autograd import Variable
import cv2





def format_img_input(img_path):
    """
    shapes image to be approrpiate size and shape for textcraft implementation - assumes gpu
    Args:
        img_path: where is image coming from
        is_cuda: are you added to GPU, will be removed

    Returns:

    """
    image=cv2.imread(img_path,cv2.IMREAD_COLOR).astype(np.float32)
    long_size=1280
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
    image, long_size, interpolation=cv2.INTER_LINEAR)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    

    return x


def resize_aspect_ratio(img, long_size, interpolation):
    """
    just resizes to ratio in conv - modification from textcraft imgproc
    Args:
        img: image
        long_size: what is expected max size
        interpolation: how are we interpolating

    Returns:

    """
    height, width, channel = img.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def get_data_png_bilingual_directory(dir_path,delimiter="_", eng_id="en",jp_id="jp")->list:
    """
    aggregates the png files for prep for training
    Args:
        dir_path: path to images
        delimiter: delimiter before eng_id or jp_id
        eng_id: the id associated for english image
        jp_id: the id associated for japanese image

    Returns:
        list
    """
    #extract the data form the bilingual directory format folders
    onlyfiles:list = [f.split(delimiter)[0] for f in listdir(dir_path) 
                 if (isfile(join(dir_path, f)) & (delimiter in f))]
    onlyfiles_set:set=set(onlyfiles)
    eng_ids:list=[]
    jp_ids:list=[]
    for name in onlyfiles_set:
        eng_ids.append(name+"{}{}.png".format(delimiter,eng_id))
        jp_ids.append(name+"{}{}.png".format(delimiter,jp_id))
    
            
    
    return eng_ids,jp_ids

class DatasetImgCraftDefault(data_torch.Dataset):
    """A datahandler for imagecraft unet architecture, simplest for tuning
    """
    def __init__(self,original_model:object,dir_path:str,eng_delimiter:str="en",jp_delimiter:str="jp"):
        self.dir_path=dir_path
        self._eng_img_names=[]
        self._jp_img_names=[]
        self._model=original_model
        self._is_cuda=next(original_model.parameters()).is_cuda
        self._is_cuda=True 
        
        self._cache={} ##cache results until I'm capping memory, throw out examples
        self._eng_img_names,self._jp_img_names=get_data_png_bilingual_directory(dir_path,"_",eng_delimiter,jp_delimiter)
        assert len(self._eng_img_names)==len(self._jp_img_names)

    
    def __len__(self):
        """
        the total number of samples
        """
        return len(self._eng_img_names)
    
 
        
    
    def __getitem__(self,index):
        if index in self._cache:
            return self._cache[index]
        
        eng_name:str=self._eng_img_names[index]
        jp_name:str=self._jp_img_names[index]
            
        eng_path:str=os.path.join(self.dir_path,eng_name)
        jp_path:str=os.path.join(self.dir_path,jp_name)
            
        torch_eng=None
        torch_jp=None
        
        
        
        
        torch_eng:torch.Tensor=format_img_input(eng_path)
        torch_jp:torch.Tensor=format_img_input(jp_path)
        
        if self._is_cuda:
            torch_eng=torch_eng.cuda()
            torch_jp=torch_jp.cuda()
        
        y:torch.Tensor=None
        x:torch.Tensor=torch_jp
        feature:torch.Tensor=None
        with torch.no_grad():
            y, feature = self._model(torch_eng) #ignore the feature maps
            del torch_eng
        #del x
        #del y
    
        self._cache[index]=(x.to("cpu"),y.to("cpu"),feature.to("cpu"))
        
        return x,y,feature
       