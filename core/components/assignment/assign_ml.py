import sys
sys.path.append("../../../")
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pandas as pd
import cv2
from core.components.assignment import assign_text
from core.training.feature_engineering import traditional_feature_prediction
from core.training.feature_engineering.traditional_feature_prediction import FeaturePredictionTraditional
from core.training.feature_engineering.iou_prediction import PredictionBoundingTraditional

class AssignTextML(assign_text.AssignDefault):
    """
    Attributes:
        _model_font_size: a model for estimating font size, this is typically set via set_font_model
        _model_location: a model for determining where to center and size the boudning box, typically set via set_location_model
    """
    def __init__(self):
        """
            extension of assign ml that incorporates models for predicting bounding boxes and font size
        """
        super().__init__()
        self._model_font_size:FeaturePredictionTraditional=None
        self._model_location:PredictionBoundingTraditional=None



    def set_font_model(self,model):
        """
        sets the font model
        Args:
            model: model to be passed, typically FeaturePredictionTraditional

        Returns:
            None
        """
        self._model_font_size=model
    
    def set_location_model(self,model):
        """
        sets the location model
        Args:
            model: model being passed, typicall iou_prediction

        Returns:
            None
        """
        self._model_location=model
    
    
    def assign_all(self,image_cv:np.array,texts:list,data:pd.DataFrame,font_path:str)->np.array:
        """
            assigns text to bounding locations using limited heuristics
        Args:
            image_cv: a 3 channel numeric array representing the image
            texts: text to be assigned to the image area
            data: a formated dataframe with features to be used for setting bounding areas
            font_path: a path to a font src to write text

        Returns:
            np.array
        """
        self._estimated_sizes=[]
        image = Image.fromarray(image_cv)
        draw = ImageDraw.Draw(image)
        font_sizes_pred:list=self._model_font_size.predict(data[self._model_font_size._x_names],True).astype(int)
        box_predictions:list=self._model_location.predict(data[self._model_location._x_names],True).astype(int)

        for text,font_size,box in zip(texts,font_sizes_pred,box_predictions):

            xmin,ymin,xmax,ymax=box[0],box[1],box[2],box[3]
            font = ImageFont.truetype(font_path, font_size)
            realigned_text = assign_text.text_wrap(text, font, xmax-xmin)

            updated_font_size = assign_text.calc_font_size(xmin, xmax, ymin, ymax, "\n".join(realigned_text))

            updated_font_size=int((font_size*.5+updated_font_size*.5))
            self._estimated_sizes.append(updated_font_size)
            font = ImageFont.truetype(font_path, updated_font_size)
            draw.text([xmin+3,ymin], "\n".join(realigned_text), font=font, fill=(0, 0, 0, 255))

        return np.asarray(image)
    

def load_default_model(model_font_pth="core/training/feature_engineering/temp1.pkl",
                       model_box_pth="core/training/feature_engineering/temp2.pkl"):
    """
    function for quickly loading a model with predefined paths
    Args:
        model_font_pth: path to font model
        model_box_pth: path to location model

    Returns:

    """

    m1=traditional_feature_prediction.load((model_font_pth))
    m2=traditional_feature_prediction.load((model_box_pth))

    aML=AssignTextML()
    aML.set_font_model(m1)
    aML.set_location_model(m2)
    return aML

if __name__ == '__main__':
    pass
    import os

    img_pth:str="C:\\Users\\egasy\\Downloads\\ComicTransferSuper\\ComicTransfer\\data\\temp_image.png"
    results_path:str="C:\\Users\\egasy\\Downloads\\ComicTransferSuper\\ComicTransfer\\data\\sample_bounding_results.tsv"
    actual_font_path:str="C://Users//egasy//Downloads//liberation-mono//LiberationMono-Bold.ttf"
    image = cv2.imread(img_pth, cv2.IMREAD_COLOR)

    print(type(image))
    actual_results_pd:str=pd.read_csv(results_path,sep="\t",index_col=0) #results typically from another model


    model_font_pth:str="C:\\Users\\egasy\\Downloads\\ComicTransferSuper\\ComicTransfer\\core\\models\\temp1.pkl"
    model_box_pth:str="C:\\Users\\egasy\\Downloads\\ComicTransferSuper\\ComicTransfer\\core\\models\\temp2.pkl"

    m1=traditional_feature_prediction.load((model_font_pth))
    m2=traditional_feature_prediction.load((model_box_pth))

    font1=m1.predict(actual_results_pd[m1._x_names].values,True).astype(int)
    loc1=m2.predict(actual_results_pd[m2._x_names].values,True).astype(int)

    aML=AssignTextML()
    aML.set_font_model(m1)
    aML.set_location_model(m2)
    result_data=aML.assign_all(image,actual_results_pd.translation.values,actual_results_pd, actual_font_path)


    image_to_draw = Image.fromarray(result_data)
    image_to_draw.save("test1.png")

    print("OK")
    #load image
    #cv read image
    #load results
    #load model1
    #load model2

    #predict font1
    #predict boxes


#AssignTextML()