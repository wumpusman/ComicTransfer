import pytest
import pandas as pd
import sys
from numpy import testing as np_test
import numpy as np


from core.datahandling import process_bilingual_data
from core.models import traditional_feature_prediction
from sklearn.linear_model import LinearRegression


@pytest.fixture
def get_text_processing_obj():
    """
    create a process object that is used for handling data for training and prediction
    Returns:
        process_bilingual_data.Preprocess_Bilingual
    """

    data_path: str = "data/sample_train/sample_data.tsv"
    all_manga = pd.read_csv(data_path, sep="\t", index_col=0)
    all_manga = all_manga.drop(columns=["level_0"])
    pre_bi = process_bilingual_data.Preprocess_Bilingual()
    pre_bi.set_data((all_manga))

    return pre_bi


@pytest.fixture
def get_basic_model():
    """
    create a wrapper for training and predictions object
    Returns:
        process_bilingual_data.FeaturePredictionTraditional
    """

    model = traditional_feature_prediction.FeaturePredictionTraditional()
    model.set_model(LinearRegression())  # create the simplest model

    return model


def test_font_prediction_train(get_text_processing_obj, get_basic_model):
    x_features_names = ["width_jp", "height_jp",
                        "text_jp_len"]  # features we wish to evaluate
    y_features_names = ["font-size_en"]  # feature to predict

    pre_bi = get_text_processing_obj
    font_model_wrapper = get_basic_model

    all_data = pre_bi.output_all_features()
    # set data frame with all possible features we'd like to examine
    font_model_wrapper.set_data(all_data)

    font_model_wrapper.set_features(x_features_names, y_features_names)

    tr_scores, dev_scores = font_model_wrapper.score_cv()  # cross validation results

    print(dev_scores)
    # expected output
    np_test.assert_allclose(
        tr_scores, 2.0, .5), "training on small subset of data with linear model, training results are different for font size"
    np_test.assert_allclose(
        dev_scores, 1.5, .75), "dev on small subset of data with linear model, training results are different for font size"

    font_model_wrapper.fit(
        font_model_wrapper._x,
        font_model_wrapper._y,
        preprocess=True)

    results_good = font_model_wrapper.predict(
        font_model_wrapper._x, preprocess=True)
    results_bad = font_model_wrapper.predict(
        font_model_wrapper._x,
        preprocess=False)  # improperly normalized data

    np_test.assert_almost_equal(
        results_good[0], 15, .5), "font size output has changed, check if preprocessing has changed"
    assert results_bad[0] > 50, "unscaled values should lead to outsized values for prediction, this is not the case"
