
from core.datahandling import process_bilingual_data
from core.models import iou_prediction
import argparse
import os
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import sklearn




default_path_tsv= "../../data/bilingual_tsv"
results_save_path="/tmp/simple_evaluation"
max_time=200
per_model_time=30

parser = argparse.ArgumentParser(description='setup for training models for bounding using automl')
parser.add_argument("-d","--datadir",help="path to tsv files",default=default_path_tsv)
parser.add_argument("-s","--savepath",help="where to save outputs",default=results_save_path)
parser.add_argument("-t","--maxtime",help="max time to run in general",default=max_time)
parser.add_argument("-p","--maxtime_model",help="max time per model in seconds",default=per_model_time)




def main (dir_path:str,save_path:str,max_run_time:int,max_run_per_model:int):
    """
    runs through automl sklearn for hyperparameter and model search
    Args:
        data_dir: path to read tsv files
        save_path: path to save output files of automl
        max_run_time: how long to run it overmodel for
        max_run_per_model: max time per model

    Returns:
        None
    """
    files: list = os.listdir(dir_path)
    file_names: list = [i for i in dir_path if ("selenium.tsv" in i)]
    full_path: str = [os.path.join(dir_path, name) for name in files]

    mangas = process_bilingual_data.read_tsv_files(full_path)
    process = process_bilingual_data.Preprocess_Bilingual()
    process.set_data(mangas[0:2000])
    print("GREAT")
    x_pd, y_pd = process.output_all_features_iou()
    x = x_pd.values
    y = y_pd.values
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=max_run_time,
        per_run_time_limit=max_run_per_model,
        tmp_folder=save_path+"_temp",
        output_folder=save_path,
    )
    final_model = automl.fit(x_train, y_train, dataset_name="tsv of manga for iou prediction")

    y_test_pred = final_model.predict(x_test)
    total_score=0
    for y_pred_el,y_test_el in zip(y_test_pred,y_test):
        total_score+= iou_prediction.get_iou_lists(y_pred_el, y_test_el)

    print(total_score/len(y_test))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.datadir, args.savepath, args.maxtime, args.maxtime_model)

