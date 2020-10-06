from core.components.translation import predictor_translate
from core.components.assignment import assign_ml
from core.components.clean import clean_img
from core.components.alignment import predict_jp_bounding
from core.training.feature_engineering.traditional_feature_prediction import FeaturePredictionTraditional
from core.training.feature_engineering.iou_prediction import PredictionBoundingTraditional
import pandas as pd
import numpy as np
import pandas as pd
import os
import os
import io
import PIL.Image as Image
import cv2

class PipeComponents():

    def __init__(self,project_id="typegan",font_path:str=""):
        self._default_font_path=font_path
        self.extract_boundary_obj=predict_jp_bounding.BoundingGoogle()
        self.clean_img_obj=clean_img.CleanDefault()
        self.translate_obj=predictor_translate.TranslationGoogle(project_id)
        self.assign_obj=assign_ml.AssignTextML()

        self._image_unprocessed=None
        self._image_cleaned=None #original image removed
        self._data_estimates=None #font predictions and text assignment estimates if they exist
        self._image_text_mask=None #image with just the text in the estimated location
        self._image_overlaid_text=None #image with text overlaid on it

        if font_path == "":
            self._default_font_path='/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'

        self._ran_once=False #did it run once


    def set_boundary_estimate_model(self,model):
        self.extract_boundary_obj=model
    def set_clean_model(self,model):
        self.clean_img_obj=model
    def set_translate_model(self,model):
        self.translate_obj=model
    def set_assignment_model(self,model):
        self.assign_obj=model

    def has_run(self):
        return self._ran_once

    def clear_prev_estimates(self):
        self._ran_once=False
        self._image_unprocessed=None
        self._image_cleaned=None
        self._data_estimates=None
        self._image_text_mask=None
        self._image_overlaid_text=None

    def calculate_results_from_path(self,img_path:str,original_text:list=[]):
        content = None
        with io.open(img_path, 'rb') as image_file:
            content = image_file

            self.calculate_results(content,original_text)


    def calculate_results(self,bytestream, original_text:list=[]):
        self.clear_prev_estimates() ## clear results
        results_bounds_ocr:dict=self.extract_boundary_obj.predict_byte(bytestream)
        formatted_bounds_ocr:pd.DataFrame=self.extract_boundary_obj.format_predictions(results_bounds_ocr)

        original_jp_text:list=formatted_bounds_ocr["text_jp"]
        translation_en=self.translate_obj.predict(original_jp_text)

        image = Image.open(bytestream)
        self._image_unprocessed = np.copy(np.asarray(image))
        self._image_cleaned=self.clean_img_obj.clean(np.copy(self._image_unprocessed),results_bounds_ocr["vertices"])

        self._image_text_mask=np.copy(np.asarray(self._image_cleaned))
        self._image_text_mask[:,:,:]=0
        self._image_text_mask=self.assign_obj.assign_all(self._image_text_mask,
                                                             translation_en,formatted_bounds_ocr,
                                                             self._default_font_path)

        self._image_overlaid_text=self.assign_obj.assign_all(self._image_cleaned,
                                                             translation_en,formatted_bounds_ocr,
                                                             self._default_font_path)

        self._data_estimates=pd.DataFrame()
        self._data_estimates["jp_text"]=original_jp_text
        self._data_estimates["en_trans"]=translation_en
        self._data_estimates["font_prediction"]=self.assign_obj.get_estimate_font_size()
        self._ran_once=True



if __name__ == '__main__':
        image_path="../../data/007.png"
        image_cv=cv2.imread(image_path,cv2.IMREAD_COLOR)

        pipeline_obj=PipeComponents()

        model_text_pth="../training/feature_engineering/temp2.pkl"
        model_font_size_pth="../training/feature_engineering/temp1.pkl"
        assign_obj=assign_ml.load_default_model(model_font_size_pth,model_text_pth)
        pipeline_obj.set_assignment_model(assign_obj)

        print("OK")
        pipeline_obj.calculate_results_from_path(image_path)

        print("done_temp")
        #pipeline_obj.calculate_results()
















