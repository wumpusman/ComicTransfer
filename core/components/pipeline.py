from core.components.translation import predictor_translate
from core.components.assignment import assign_ml
from core.components.clean import clean_img
from core.components.alignment import predict_jp_bounding
import numpy as np
import pandas as pd
import io
import PIL.Image as Image
import cv2

class PipeComponents():
    """
    Attributes:
        _default_font_path: path to font
        extract_boundary_obj: object handling boundary detection
        clean_img_obj: object handling cleaning
        translate_obj: object handling translation
        self.assign_obj: object handling assignment
        _image_unprocessed: raw image to be processed
        _image_cleaned: image after cleaning
        _data_estimates: data collected after running models
        _image_text_mask: mask with just reformatted text on it
        _image_overlaid_text: image with overlaid text
        _ran_once: did this system go through dataset at least once

    """
    def __init__(self,project_id="typegan",font_path:str=""):
        """
        pipeline that is meant to encapsulate the different aspects of the current system
        Args:
            project_id: id used for project certification
            font_path: path to a specific font to be rendered, default is rooted to linux path
        """
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
        self._ran_once = False  # did it run once
        if font_path == "":
            self._default_font_path= '../../data/LiberationMono-Bold.ttf'




    def set_boundary_estimate_model(self,model):
        """
        sets boundary object
        Args:
            model: model to be based from components, associated with alignment package
        Returns:
            None

        """
        self.extract_boundary_obj=model
    def set_clean_model(self,model):
        """
        sets clean object
        Args:
            model: model to be based from components, associated with clean package
        Returns:
            None

        """
        self.clean_img_obj=model
    def set_translate_model(self,model):
        """
        sets translate object
        Args:
            model: model to be based from components, associated with translation package
        Returns:
            None

        """
        self.translate_obj=model
    def set_assignment_model(self,model):
        """
        sets assignment object
        Args:
            model: model to be based from components, associated with assignment package
        Returns:
            None

        """
        self.assign_obj=model

    def has_run(self)->bool:
        """
        returns if pipeline has run at least once
        Returns:
            bool
        """
        return self._ran_once

    def clear_prev_estimates(self):
        """
        explicitely clear out images and results
        Returns:
            None
        """
        self._ran_once=False
        self._image_unprocessed=None
        self._image_cleaned=None
        self._data_estimates=None
        self._image_text_mask=None
        self._image_overlaid_text=None

    def calculate_results_from_path(self,img_path:str,original_text:list=[]):
        """
        calculates results of pipeline based on an image path and optional associated transcript
        results are stored internally
        Args:
            img_path: image path
            original_text: orignial transcripts if avalailable

        Returns:
            None
        """
        content = None
        with io.open(img_path, 'rb') as image_file:
            content = image_file

            self.calculate_results(content,original_text)


    def calculate_results(self,bytestream, original_text:list=[]):
        """ calculates results of pipeline based on an image bytestream and optional associated transcript
        results are stored internally

        Args:
            bytestream: bytestream of image
            original_text: original transcripts if avalailable

        Returns:
            None
        """
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



















