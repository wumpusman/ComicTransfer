
import os
import sys
sys.path.append("../../")
from core.datahandling import process_bilingual_data
from core.models import traditional_feature_prediction, iou_prediction
from sklearn.linear_model import LinearRegression
import argparse

default_path_tsv= "../../data/bilingual_tsv"
save_model_path= "feature_engineering/temp2.pkl"
save_model:bool=True
run_ablation:bool=False
model_type:str="bound"

parser = argparse.ArgumentParser(description='setup for training models, as well as simple ablations')
parser.add_argument("-d","--datadir",help="path to tsv files",default=default_path_tsv)
parser.add_argument("-m","--savepath",help="path to save the model",default=save_model_path)
parser.add_argument("-s","--savemodel",help="save the model",default=save_model)
parser.add_argument("-a","--ablate",help="run and output simple ablations",default=run_ablation)
parser.add_argument("-t","--type",help="select 'font' or 'bound' to decide what model to run",default=model_type)




def main (model_type:str,dir_path,save_path:str="temp.pkl",save_model:bool=False,run_ablation:bool=False):
    """
    runs basic model training, and simple ablation if using non-linear models
    Args:
        model_type: is the model for predicting bounds or font size
        dir_path: the directory where underlying data is used for trianing
        save_path: where is the model going to be saved
        save_model: should the model be saved
        run_ablation: do you want to run ablations on each feature

    Returns:
        None
    """

    files: list = os.listdir(dir_path)
    file_names: list = [i for i in dir_path if ("selenium.tsv" in i)]
    full_path: str = [os.path.join(dir_path, name) for name in files]
    print("preparing data")
    mangas = process_bilingual_data.read_tsv_files(full_path)
    process = process_bilingual_data.Preprocess_Bilingual()
    process.set_data(mangas)
    all_data = process.output_all_features()

    print("training")
    #["x1_jp","y1_jp","x2_jp","y2_jp","text_jp_len",'left_jp','width_jp','height_jp']
    x_pd, y_pd, x_names, y_names = None, None, None, None
    prediction_wrapper = None
    if model_type=="bound":
        x_pd, y_pd = process.output_all_features_iou()
        x_names = x_pd.columns.values
        y_names = y_pd.columns.values

        all_data = process.output_all_features()
        prediction_wrapper = iou_prediction.PredictionBoundingTraditional()
        prediction_wrapper.set_data(all_data)
        prediction_wrapper.set_features(x_names)
        prediction_wrapper.set_features(x_names,y_names)

    elif model_type=="font":
        x_pd, y_pd = process.output_all_features_font_size()
        x_names = x_pd.columns.values
        y_names = y_pd.columns.values

        prediction_wrapper = traditional_feature_prediction.FeaturePredictionTraditional()
        prediction_wrapper.set_data(all_data)
        prediction_wrapper.set_features(x_names,y_names)

    else:
        raise Exception("no known model type specified")

    #MultiOutputRegressor(GradientBoostingRegressor())
    prediction_wrapper.set_model(LinearRegression())
    print(prediction_wrapper.score_cv())


    if run_ablation:
        for feature in range(len(x_names)):
            print("{} feature removed".format(x_names[feature]))
            temp_x_names = x_names.copy().tolist()
            del temp_x_names[feature]
            prediction_wrapper.set_features(temp_x_names)
            print(prediction_wrapper.score_cv())

    if save_model:

        prediction_wrapper.fit(prediction_wrapper._x, prediction_wrapper._y,preprocess=True)
        traditional_feature_prediction.save(prediction_wrapper, save_path)


if __name__ == '__main__':


    args = parser.parse_args()

    main(args.type,args.datadir, args.savepath, args.savemodel, args.ablate)


