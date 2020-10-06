from core.datahandling import process_bilingual_data
from core.training.feature_engineering import iou_prediction
import os
import sys

import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import sklearn

if __name__ == '__main__':
    path= "../../../data/bilingual_tsv"
    files:list=os.listdir(path)
    file_names:list=[i for i in path if ("selenium.tsv" in i)]
    full_path:str=[os.path.join(path,name) for name in files]

    #bilingual_handler=process_bilingual_data.Preprocess_Bilingual()

    mangas=process_bilingual_data.read_tsv_files(full_path)


    process=process_bilingual_data.Preprocess_Bilingual()
    process.set_data(mangas[0:10000])
    box_area = process.extract_box_area(False)
    box_location = process.extract_box_location(False)
    box_coords = process.to_box_coords(False)
    text_len = process.extract_text_length()
    font_size = process.extract_font_size()

    aggregated=process.aggregate_to_pandas([box_area,box_location,box_coords,text_len,font_size])
    x_names = ["x1_jp","y1_jp","x2_jp","y2_jp","top_jp", "left_jp", "width_jp", "height_jp","text_jp_len"] #,"y21","y22","x22","x12"]
    y_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']



    print(mangas.shape)

    x=aggregated[x_names].values
    y=aggregated[y_names].values
    import autosklearn.regression
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x, y, random_state=1)


    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=1000,
        per_run_time_limit=20,
        tmp_folder='/tmp/autosklearn_regression_example_tmp',
        output_folder='/tmp/autosklearn_regression_example_out',
    )

    final_model=automl.fit(x_train, y_train, dataset_name="just_testing_it_out")
    print("OK")

    y_test_pred = final_model.predict(x_test)


    from core.training.feature_engineering import iou_prediction
    total_score=0
    for y_pred_el,y_test_el in zip(y_test_pred,y_test):
        total_score+=iou_prediction.get_iou_lists(y_pred_el,y_test_el)

    print(total_score/len(y_test))