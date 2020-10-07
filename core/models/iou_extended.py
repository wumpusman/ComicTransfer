import pandas as pd
from core.models import iou_prediction


class PredictionBoundingSeparate(iou_prediction.PredictionBoundingTraditional):
    """
    Predicts bounding but estimates x,y, width and height separately as opposed to x,y,x1,y2
    """

    def __init__(self):
        super().__init__()


    def _transform_to_coord(self,data):
        results_pd= pd.DataFrame(data)
        transformed=pd.DataFrame()

        transformed["x1"]=results_pd[0]

        transformed["y1"]=results_pd[1]
        transformed["x2"]=results_pd[2]+results_pd[0]
        transformed["y2"]=results_pd[3]+results_pd[1]
        return transformed.values

    def score(self,predicted,ground_truth) ->float:
        predicted=self._transform_to_coord(predicted)
        ground_truth=self._transform_to_coord(ground_truth)
        return super().score(predicted,ground_truth)


    def set_features(self, x_names: list = ["top_jp", "left_jp", "width_jp", "height_jp", "text_jp_len"],y_names: list = ["left_en",'top_en',"width_en","height_en"]):
        super().set_features(x_names, y_names)