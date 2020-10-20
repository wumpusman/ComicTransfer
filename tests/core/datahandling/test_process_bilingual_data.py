import pytest
import pandas as pd
import sys


from core.datahandling import process_bilingual_data


@pytest.fixture
def get_default_object():
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


def test_set_data():
    """
    ensure that the data is properly extracted, and the basic feature names have been extracted
    Returns:

    """
    data_path: str = "data/sample_train/sample_data.tsv"
    all_manga = pd.read_csv(data_path, sep="\t", index_col=0)
    all_manga = all_manga.drop(columns=["level_0"])
    pre_bi = process_bilingual_data.Preprocess_Bilingual()
    pre_bi.set_data(all_manga)
    print(all_manga.columns)
    print(all_manga.shape)

    # expected original size
    assert all_manga.shape == (313, 22)
    assert pre_bi._data_to_process.shape == (157, 43)

    pre_bi_columns = pre_bi._data_to_process.columns.tolist()

    # ensure add meta features are there
    assert "nn1" in pre_bi_columns, "nearest neighbor feature name does not exist, meta_features may have been modified"

    assert "link" in pre_bi_columns, "link feature name does not exist, data format may have changed"
    # ensure basic featurs were added
    assert "text_jp" in pre_bi_columns, "text english feature not defined after formatting"
    assert "text_en" in pre_bi_columns, "text japanese feature not defined after formatting"


def test_feature_extraction(get_default_object):
    """
    ensure that the features expected still exist and data pipeline has not been modified
    """

    pre_bi = get_default_object

    temp = pre_bi.extract_text()
    assert temp['text_en'][0] == "AH, THAT'S AWESOME!", "text feature missing or has been modified"

    # test three
    temp = pre_bi.extract_text_length()
    assert temp["text_jp_len"][0] == 8, "text len no longer matches expected size"
    assert temp["text_en_len"][1] == 13, "text len no longer matches expected size"

    # test four
    temp = pre_bi.extract_fonts()
    assert temp["font-family_jp"][0] == ' yasashisa-antique, sans-serif', "font feature does not match expected output"

    # test five
    temp = pre_bi.extract_font_size()
    assert temp["font-size_jp"][0] == 20, "font size processing have been modified"

    # test six
    temp = pre_bi.extract_box_area(normalize=False)
    assert temp["width_jp"][0] == 64, "width no longer corresponding to expected size"
    temp = pre_bi.extract_box_area(normalize=True)
    assert temp["width_jp"][0] == .08, "normalized size does not match"

    # test seven
    temp = pre_bi.extract_box_location(normalize=True)
    assert temp["left_jp"][4] == .04250, "normalized coordinate position no longer matches"
    temp = pre_bi.extract_box_location(normalize=False)
    assert temp["left_jp"][4] == 34, "raw coordinate position no longer matches expected output"

    # test eight
    temp1 = pre_bi.extract_box_location(normalize=True)
    temp2 = pre_bi.extract_font_size()
    temp3 = pre_bi.extract_text_length()
    joined_temp = pre_bi.aggregate_to_pandas([temp1, temp2, temp3])
    assert joined_temp.shape == (
        157, 8), "feature size has changed, has extract font, text, or box extraction changed"

    # get the right bounding boxes
    temp = pre_bi.to_box_coords(False)
    assert (int(temp["x2_jp"][2]), int(temp["y2_jp"][2])) == (
        350, 1069), "remapping of coordinates does not match expected shape for unnormalized position"


def test_meta_feature_extraction(get_default_object):
    """
        ensure that the types of meta features have not changed in terms of extraction
    """
    pre_bi = get_default_object

    temp = pre_bi.extract_macro_features()

    assert (int(temp["nn1"][0]) ==
            309), "nearest neighbor value does not match expected"
    assert (int(temp["nn2"][0]) ==
            667), "2nd nearest neighbor distance does not match expected"
    assert (int(temp["box_num"][0]) ==
            6), "number of boxes on page does not match expected"

    temp = pd.DataFrame((pre_bi.extract_squared_features(False)))
    assert(int(temp["x1_jp_squared"][0]) == (410881)
           ), "squared x1 coord feature does not match expected"
    assert(int(temp["height_jp_squared"][1] == (5184))
           ), "squared height does not match expected"


def test_feature_prepare(get_default_object):
    """
    ensures that the macro features expected to be extracted have not been modified
    """
    pre_bi = get_default_object

    temp = pre_bi.output_all_features()

    x_feature_names = ['width_jp', 'height_jp', 'top_jp', 'left_jp',
                       'x1_jp', 'y1_jp', 'x2_jp', 'y2_jp', 'text_jp_len',
                       'nn1', 'nn2', 'box_num', 'x1_jp_squared',
                       'y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared',
                       'width_jp_squared']

    x, y = pre_bi.output_all_features_font_size()
    assert (
        x.shape == (
            157, 17)), "expected feature list for font_size  prediction is different from what was expected"
    assert (
        y.shape) == (
        157, 1), "expected output for font_size is different from what was expected"

    assert x_feature_names == x.columns.tolist(
    ), "expected feature list has changed for font feature creation"

    x, y = pre_bi.output_all_features_iou()

    x_feature_names = ['width_jp', 'height_jp', 'top_jp', 'left_jp',
                       'x1_jp', 'y1_jp', 'x2_jp', 'y2_jp', 'text_jp_len',
                       'nn1', 'nn2', 'box_num', 'x1_jp_squared',
                       'y1_jp_squared', 'x2_jp_squared', 'y2_jp_squared',
                       'width_jp_squared']

    assert (
        x.shape == (
            157, 17)), "expected feature list for bounding  prediction is different from what was expected"
    assert (
        y.shape) == (
        157, 4), "expected output for bounding_prediction is different from what was expected"

    assert x_feature_names == x.columns.tolist(
    ), "expected feature list has changed for font feature creation"
