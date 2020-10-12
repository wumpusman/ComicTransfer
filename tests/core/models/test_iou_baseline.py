import pytest
import pandas as pd
import sys
from numpy import testing as np_test
import numpy as np
import os
#


from core.datahandling import process_bilingual_data
from core.models import iou_basline
from core.models import iou_prediction
from sklearn.linear_model import LinearRegression


@pytest.fixture
def get_text_processing_obj():
    """
    create a process object that is used for handling data for training and prediction
    Returns:
        process_bilingual_data.Preprocess_Bilingual
    """

    data_path: str = "data/sample_data.tsv"
    all_manga = pd.read_csv(data_path, sep="\t", index_col=0)
    all_manga = all_manga.drop(columns=["level_0"])
    pre_bi = process_bilingual_data.Preprocess_Bilingual()
    pre_bi.set_data((all_manga))

    return pre_bi


@pytest.fixture
def get_basic_model():
    """
    create a wrapper for training and predictions object if you were to do really simplistic naive approach
    Returns:
        iou_prediction.PredictionBoundingBaseline
    """

    model = iou_basline.PredictionBoundingBaseline()
    model.set_model(LinearRegression())  # create the simplest model

    return model


def test_bounding_prediction_train(get_text_processing_obj, get_basic_model):
    """
    evalute the basic bounding box training and prediction
    Args:
        get_text_processing_obj: object for processing data
        get_basic_model: object for handling training and prediction of bounding areas of text

    Returns:
        None

    """

    x_features_names = ["x1_jp", "y1_jp", "x2_jp",
                        "y2_jp"]  # features we wish to evaluate
    y_features_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']

    pre_bi = get_text_processing_obj
    bound_model_predictor = get_basic_model

    all_data = pre_bi.output_all_features()
    print(all_data.columns)
    # set data frame with all possible features we'd like to examine
    bound_model_predictor.set_data(all_data)

    bound_model_predictor.set_features()  # basic names should just be

    np_test.assert_equal(x_features_names, bound_model_predictor._x_names)

    bound_model_predictor.fit(
        bound_model_predictor._x_names,
        bound_model_predictor._y_names)

    # expected output
    tr_scores, dev_scores = bound_model_predictor.score_cv()  # cross validation results

    np_test.assert_allclose(
        tr_scores, .65, .05), "baseline with almost no data, training results are different for bounding"
    np_test.assert_allclose(
        dev_scores, .63, .05), "baseline dev with almost no data , training results are different for bounding"
