import pytest
import pandas as pd
import sys
from numpy import testing as np_test
import numpy as np

sys.path.append("../../..")
from core.datahandling import process_bilingual_data
from core.models import traditional_feature_prediction
from core.models import iou_prediction
from sklearn.linear_model import LinearRegression
@pytest.fixture
def get_text_processing_obj():
    """
    create a process object that is used for handling data for training and prediction
    Returns:
        process_bilingual_data.Preprocess_Bilingual
    """

    data_path: str = "../../../data/sample_data.tsv"
    all_manga = pd.read_csv(data_path, sep="\t", index_col=0)
    all_manga = all_manga.drop(columns=["level_0"])
    pre_bi=process_bilingual_data.Preprocess_Bilingual()
    pre_bi.set_data((all_manga))


    return pre_bi

@pytest.fixture
def get_basic_model():
    """
    create a wrapper for training and predictions object
    Returns:
        iou_prediction.PredictionBoundingTraditional
    """


    model = iou_prediction.PredictionBoundingTraditional()
    model.set_model(LinearRegression()) #create the simplest model

    return model



def test_bounding_prediction_train(get_text_processing_obj,get_basic_model):
    """
    evalute the basic bounding box training and prediction
    Args:
        get_text_processing_obj: object for processing data
        get_basic_model: object for handling training and prediction of bounding areas of text

    Returns:
        None

    """

    x_features_names = ["x1_jp","y1_jp","x2_jp","y2_jp","top_jp","text_jp_len"]  # features we wish to evaluate
    y_features_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']

    pre_bi=get_text_processing_obj
    bound_model_predictor=get_basic_model

    all_data= pre_bi.output_all_features()
    print(all_data.columns)
    bound_model_predictor.set_data(all_data) #set data frame with all possible features we'd like to examine

    bound_model_predictor.set_features(x_features_names,y_features_names)

    tr_scores, dev_scores = bound_model_predictor.score_cv() # cross validation results


    #expected output


    np_test.assert_allclose(tr_scores,.7,.05), "training on small subset of data with linear model, training results are different for bounding"
    np_test.assert_allclose(dev_scores, .675,.07), "dev on small subset of data with linear model, training results are different for bounding"


    bound_model_predictor.fit(bound_model_predictor._x, bound_model_predictor._y, preprocess=True)

    results_good=bound_model_predictor.predict(bound_model_predictor._x, preprocess=True)
    results_bad=bound_model_predictor.predict(bound_model_predictor._x, preprocess=False) #improperly normalized data



    assert np.abs(results_good[0][0]-610) <70, "bounding size for x1 has changed, check if preprocessing has changed"
    assert results_bad[0][0] > 13000, "unscaled values expected to be larger, this is not the case, check if preprocessing or feature training has changed"



