import sys
sys.path.append("../..")
from core.models.traditional_feature_prediction import FeaturePredictionTraditional


def get_iou(bb1: dict, bb2: dict) -> float:
    """
        an implementation of Intersection over union slow - very slightly modified from #todo optimize it
        https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Args:
        bb1: The (x1, y1) (x2, y2) position is at the top left corner, bottom right corner
        bb2: The (x1, y1) (x2, y2) position is at the top left corner, bottom right corner

    Returns:
        float
    """

    try:
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']
    except:
        return 0  # if predictions are off return failure

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_lists(l1: list, l2: list) -> float:
    """
    does iou lists instead of raw components
    Args:
        l1: list of values x1,y1,x2,y2
        l2: list of values x1,y1,x2,y2

    Returns:
        float
    """
    bb1 = {"x1": l1[0], "y1": l1[1], "x2": l1[2], "y2": l1[3]}
    bb2 = {"x1": l2[0], "y1": l2[1], "x2": l2[2], "y2": l2[3]}

    return get_iou(bb1, bb2)


class PredictionBoundingTraditional(FeaturePredictionTraditional):

    def __init__(self):
        """
        Same as prediction bounding, but uses IOU metric and assumes output of the form
        x1,y1 (top left), x2,y2 (bottom right)
        """

        super().__init__()

    def set_features(
        self,
        x_names: list = [
            "top_jp",
            "left_jp",
            "width_jp",
            "height_jp",
            "text_jp_len"],
        y_names: list = [
            'x1_en',
            'y1_en',
            'x2_en',
            'y2_en']):
        """
        sets features with the one expectation that output y features are x1,y1,x2,y2 format
        Args:
            x_names: input feature list of names
            y_names: output features to be predicted, it's expected to be x1, y1, x2, y2 format

        Returns:
            None
        """
        super().set_features(x_names, y_names)

    def score(self, predicted, ground_truth) -> float:
        """
        score the values based on IOU, how much the two bounding boxes are overlapping
        Args:
            predicted: the predicted coordinates in the form x1,y1,x2,y2 (top left, bottom right)
            ground_truth: a 2D matrix of (N,4) where each row is a specific coordinate set of (x1, y1,x2, y2)

        Returns:
            float
        """
        total_score = 0

        for y_hat, y in zip(predicted, ground_truth):

            total_score += get_iou_lists(y_hat, y)

        return total_score / len(predicted)

    def fit(self, x, y, preprocess: bool = False):
        if preprocess:
            x = self.preprocess(x, True)
        self._model.fit(x, y)
