import pandas as pd
from core.models import iou_prediction


class PredictionBoundingSeparate(iou_prediction.PredictionBoundingTraditional):

    def __init__(self):
        """
        Predicts bounding but estimates x,y, width and height separately as opposed to x,y,x1,y2
        """
        super().__init__()

    def _transform_to_coord(self, data) -> list:
        """
            converts the output, and modifies it so that values are coordinates instead of raw features ["left_en",'top_en',"width_en","height_en"]
        Args:
            data: n

        Returns:
            list
        """
        results_pd = pd.DataFrame(data)
        transformed = pd.DataFrame()

        transformed["x1"] = results_pd[0]

        transformed["y1"] = results_pd[1]
        transformed["x2"] = results_pd[2] + results_pd[0]
        transformed["y2"] = results_pd[3] + results_pd[1]
        return transformed.values.tolist()

    def score(self, predicted, ground_truth) -> float:
        """
            calculates the iou score after transforming its output to the format of x1,y1,x2,y2
        Args:
            predicted:  predicted features "left_en",'top_en',"width_en","height_en"
            ground_truth: the actual features of the bounding boxes ["left_en",'top_en',"width_en","height_en"]

        Returns:
            float
        """
        predicted = self._transform_to_coord(predicted)
        ground_truth = self._transform_to_coord(ground_truth)
        return super().score(predicted, ground_truth)

    def set_features(
        self,
        x_names: list = [
            "top_jp",
            "left_jp",
            "width_jp",
            "height_jp",
            "text_jp_len"],
        y_names: list = [
            "left_en",
            'top_en',
            "width_en",
            "height_en"]):
        super().set_features(x_names, y_names)
