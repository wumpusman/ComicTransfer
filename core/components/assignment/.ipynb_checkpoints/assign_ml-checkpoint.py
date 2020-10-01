import sys
sys.path.append("../../../")

from core.components.assignment import assign_text
from core.training.feature_engineering.traditional_feature_prediction import FeaturePredictionTraditional
from core.training.feature_engineering.iou_prediction import PredictionBoundingTraditional
class AssignTextML(assign_text.AssignDefault):
    
    def __init__(self):
        super().__init__()
        self._model_font_size:FeaturePredictionTraditional=None
        self._model_location:PredictionBoundingTraditional=None
            
    def load_font_model(path:str):
        pass

AssignTextML()