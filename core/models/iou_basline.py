from core.models import iou_prediction

class PredictionBoundingBaseline(iou_prediction.PredictionBoundingTraditional):

    def __init__(self):
        """Overlays the JP bounding box onto the english one with no modeling
        """
        super().__init__()

    def fit(self, x, y, preprocess: bool = False):
        return None

    def predict(self, data: list, preprocess: bool = False) -> list:
        """assumes input feature shape is same output feature shape
        """
        return data

    def preprocess(self, x: list, fit_it: bool = False):
        return x

    def set_features(self):
        """Features must be the same shape and size, by default it's jp and en
        """
        x_names = ['x1_jp', 'y1_jp', 'x2_jp', 'y2_jp']
        y_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']
        super().set_features(x_names, y_names)
