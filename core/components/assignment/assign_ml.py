import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import sys
sys.path.append("../../../")
from core.models.iou_prediction import PredictionBoundingTraditional
from core.models.traditional_feature_prediction import FeaturePredictionTraditional
from core.models import traditional_feature_prediction
from core.components.assignment import assign_text


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
        self._model_font_size: FeaturePredictionTraditional = None
        self._model_location: PredictionBoundingTraditional = None

    def set_font_model(self, model):
        """
        sets the font model
        Args:
            model: model to be passed, typically FeaturePredictionTraditional

        Returns:
            None
        """
        self._model_font_size = model

    def set_location_model(self, model):
        """
        sets the location model
        Args:
            model: model being passed, typicall iou_prediction

        Returns:
            None
        """
        self._model_location = model

    def assign_all(self, image_cv: np.array, texts: list, data: pd.DataFrame,
                   font_path: str) -> np.array:
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
        self._estimated_sizes = []
        image = Image.fromarray(image_cv)
        draw = ImageDraw.Draw(image)
        font_sizes_pred: list = self._model_font_size.predict(
            data[self._model_font_size._x_names], True).astype(int)
        box_predictions: list = self._model_location.predict(
            data[self._model_location._x_names], True).astype(int)

        for text, font_size, box in zip(texts, font_sizes_pred,
                                        box_predictions):

            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
            font = ImageFont.truetype(font_path, font_size)
            realigned_text = assign_text.text_wrap(text, font, xmax - xmin)

            updated_font_size = assign_text.calc_font_size(
                xmin, xmax, ymin, ymax, "\n".join(realigned_text))

            updated_font_size = int((font_size * .5 + updated_font_size * .5))
            self._estimated_sizes.append(updated_font_size)
            font = ImageFont.truetype(font_path, updated_font_size)
            draw.text([xmin + 3, ymin],
                      "\n".join(realigned_text),
                      font=font,
                      fill=(0, 0, 0, 255))

        return np.asarray(image)


def load_default_model(model_font_pth="data/models/font_model.pkl",
                       model_box_pth="data/models/bounding_model.pkl"):
    """
    function for quickly loading a model with predefined paths
    Args:
        model_font_pth: path to font model
        model_box_pth: path to location model

    Returns:

    """

    m1 = traditional_feature_prediction.load((model_font_pth))
    m2 = traditional_feature_prediction.load((model_box_pth))

    aML = AssignTextML()
    aML.set_font_model(m1)
    aML.set_location_model(m2)
    return aML
