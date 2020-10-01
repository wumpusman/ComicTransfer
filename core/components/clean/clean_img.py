import numpy as np
import pandas as pd
class CleanDefault():
    
    
    def __init__(self):
        pass
    
    def clean(self,image_cv:np.array,bounding_boxes:[]=None,mask:np.array=None):
        """ literally add white space into the bounding boxes, super naive
        """
        for bounding in bounding_boxes:
            aligned=pd.DataFrame(bounding)
    
            xmin=aligned["x"].min()-5
            xmax=aligned["x"].max()+5
            ymax=aligned["y"].max()+5
            ymin=aligned["y"].min()-5
            image_cv[ymin:ymax,xmin:xmax,:]=255
            
        return image_cv
    
    
    def overlay_color(self,image_cv:np.array,bounding_boxes:[]):
        """ overlay color in one channel to highlight text, 
        """
        for bounding in bounding_boxes:
            aligned=pd.DataFrame(bounding)
    
            xmin=aligned["x"].min()-5
            xmax=aligned["x"].max()+5
            ymax=aligned["y"].max()+5
            ymin=aligned["y"].min()-5
            image_cv[ymin:ymax,xmin:xmax,0:1]=955
            
        return image_cv
        
    
