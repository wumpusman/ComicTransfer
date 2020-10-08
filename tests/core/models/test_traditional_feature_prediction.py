import pytest
import pandas as pd
import sys

sys.path.append("../../..")
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
        process_bilingual_data.Preprocess_Bilingual
    """


    model = traditional_feature_prediction.FeaturePredictionTraditional()
    model.set_model(LinearRegression()) #create the simplest model

    return model



def test_font_prediction_train(get_text_processing_obj,get_basic_model):
    x_features_names = ["width_jp", "height_jp", "text_jp_len"]  # features we wish to evaluate
    y_features_names = ["font-size_en"]  # feature to predict

    pre_bi=get_text_processing_obj
    font_model_wrapper=get_basic_model

    all_data= pre_bi.output_all_features()
    font_model_wrapper.set_data(all_data) #set data frame with all possible features we'd like to examine

    font_model_wrapper.set_features(x_features_names,y_features_names)

    tr_scores, dev_scores = font_model_wrapper.score_cv()

    print(tr_scores)
    print(dev_scores)
    assert False
