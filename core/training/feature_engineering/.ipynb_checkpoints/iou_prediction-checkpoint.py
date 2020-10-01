import sys
import os
sys.path.append("../../..")
from core.training.feature_engineering.traditional_feature_prediction import FeaturePredictionTraditional
from core.datahandling.process_bilingual_data import Preprocess_Bilingual
from core.training.feature_engineering import traditional_feature_prediction
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np




def get_iou(bb1:dict, bb2:dict):
    """
        an implementation of Intersection over union slow - very slightly modified from #todo optimize it
        https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    Args:
        bb1: The (x1, y1) (x2, y2) position is at the top left corner, bottom right corner
        bb2: The (x1, y1) (x2, y2) position is at the top left corner, bottom right corner

    Returns:
        float
    """
    #bb1=quick_swap(bb1,"x1","x2")
    #bb1=quick_swap(bb1, "y1", "y2")
    #bb2=quick_swap(bb2, "x1", "x2")
    #bb2=quick_swap(bb2, "y1", "y2")
    try:
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']
    except:
        return .1

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

def get_iou_lists(l1:list,l2:list)->float:
    """

    Args:
        l1: list of values x1,y1,x2,y2
        l2: list of values x1,y1,x2,y2

    Returns:
        float
    """
    bb1 =  {"x1":l1[0],"y1":l1[1],"x2":l1[2],"y2":l1[3]}
    bb2 =  {"x1": l2[0], "y1": l2[1], "x2": l2[2], "y2": l2[3]}

    return get_iou(bb1,bb2)





class PredictionBoundingTraditional(FeaturePredictionTraditional):


    def __init__(self):

        super().__init__()


    def set_features(self,x_names:list=["top_jp", "left_jp", "width_jp", "height_jp", "text_jp_len"],y_names:list=['x1_en', 'y1_en', 'x2_en', 'y2_en']):
        super().set_features(x_names,y_names)




    def score(self,predicted,ground_truth) ->float:
        total_score=0

        for y_hat,y in zip(predicted,ground_truth):

            total_score+=get_iou_lists(y_hat,y)

        return total_score/len(predicted)

    def fit(self,x,y,preprocess:bool=False):
        self._model.fit(x,y)

class PredictionBoundingBaseline(PredictionBoundingTraditional):
    """Overlays the JP bounding box onto the english one with no modeling
    """
    def __init__(self):
        super().__init__()
    
    def fit(self,x,y,preprocess:bool=False):
        return None
    
    def predict(self,data:list,preprocess:bool=False)->list:
        """assumes input feature shape is same output feature shape
        """
        return data
    
    

    def preprocess(self, x:list, fit_it:bool=False):
        return x 
    
    def set_features(self):
        """Features must be the same shape and size, by default it's jp and en
        """
        x_names=['x1_jp', 'y1_jp', 'x2_jp', 'y2_jp']
        y_names=['x1_en', 'y1_en', 'x2_en', 'y2_en']
        super().set_features(x_names,y_names)
            

class PredictionBoundingTraditional2(PredictionBoundingTraditional):
    """
    Predicts bounding but uses the x,y, width and height separately as opposed to x,y,x1,y2 
    """

    def __init__(self):
        super().__init__()


    def _transform_to_coord(self,data):
        results_pd= pd.DataFrame(data)
        transformed=pd.DataFrame()

        transformed["x1"]=results_pd[0]

        transformed["y1"]=results_pd[1]
        transformed["x2"]=results_pd[2]+results_pd[0]
        transformed["y2"]=results_pd[3]+results_pd[1]
        return transformed.values

    def score(self,predicted,ground_truth) ->float:
        predicted=self._transform_to_coord(predicted)
        ground_truth=self._transform_to_coord(ground_truth)
        return super().score(predicted,ground_truth)


    def set_features(self, x_names: list = ["top_jp", "left_jp", "width_jp", "height_jp", "text_jp_len"],y_names: list = ["left_en",'top_en',"width_en","height_en"]):
        super().set_features(x_names, y_names)

    #def preprocess(self, x:list, fit_it:bool=False):
     #   return x


        

if __name__ == '__main__':
    files_of_interest = ["Yokohama_Shopping_Trip_selenium.tsv",
                         "Toradora!_selenium.tsv",
                         "Rokuhoudou_Yotsuiro_Biyori_selenium.tsv",
                         "Kekkonshite_mo_Koishiteru_selenium.tsv"
                         ]

    all_manga_pds = []
    data_path= "/home/jupyter/ComicTransfer/data/bilingual_tsv" 
    data_name = "Doraemon_Long_Stories_selenium.tsv"

    for name in files_of_interest:
        full_path = os.path.join(data_path, name)

        all_manga = pd.read_csv(full_path, sep="\t", index_col=0)
        all_manga = all_manga.drop(columns=["level_0"])
        all_manga_pds.append(all_manga)

    all_manga=pd.concat(all_manga_pds)
    print("OK")
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor

    process = Preprocess_Bilingual()
    process.set_data(all_manga)
    process.extract_text()
    process = Preprocess_Bilingual()
    process.set_data(all_manga)

    from sklearn.multioutput import MultiOutputRegressor

    from sklearn.neighbors import KNeighborsRegressor


    box_area = process.extract_box_area()
    box_location=process.extract_box_location()
    text_info=process.extract_text()
    aggregated=process.aggregate_to_pandas([box_area,box_location,text_info])
    x_names: list = ["width_jp", "height_jp", "text_jp_len"]

    box_area = process.extract_box_area(False)
    box_location = process.extract_box_location(False)
    box_coords = process.to_box_coords(False)
    text_info = process.extract_text_length()
    font_size = process.extract_font_size()
    
    aggregated = process.aggregate_to_pandas([box_coords, box_area, box_location, text_info,font_size])
    aggregated["x12"]=aggregated["x1_jp"]*aggregated["x1_jp"]
    aggregated["x12"] = aggregated["x1_jp"] * aggregated["x1_jp"]
    aggregated["x22"]=aggregated["x2_jp"]*aggregated["x2_jp"]
    aggregated["x22"] = aggregated["x2_jp"] * aggregated["x2_jp"]
    aggregated["y22"] = aggregated["y2_jp"] * aggregated["y2_jp"]
    aggregated["y21"] = aggregated["y1_jp"] * aggregated["y1_jp"]
    
    p=PredictionBoundingBaseline()
    p.set_data(aggregated)
    p.set_features()
    
    print(aggregated.columns)
    print(p.score_cv())
    
    x_names = ["x1_jp","y1_jp","x2_jp","y2_jp","top_jp", "left_jp", "width_jp", "height_jp","text_jp_len"] #,"y21","y22","x22","x12"]
    y_names = ['x1_en', 'y1_en', 'x2_en', 'y2_en']

    #x_names =['x1_jp', 'y1_jp', 'x2_jp', 'y2_jp']#  ["y21", "y22", "x22", "x12", "x1_jp", "y1_jp", "x2_jp", "y2_jp", "top_jp", "left_jp", "width_jp",

    y_names = ["left_jp",'top_jp',"width_jp","height_jp"] #, 'y1_en', 'x2_en', 'y2_en']
    #y_names = ['x1_jp', 'y1_jp', 'x2_jp', 'y2_jp']
    #x_names = ['x1_jp', 'y1_jp', 'x2_jp', 'y2_jp'] # "text_jp_len"]
    y_names =  ["left_en",'top_en',"width_en","height_en"]#['x1_en', 'y1_en', 'x2_en', 'y2_en']
    #y_names = ["width_jp"]
    b=PredictionBoundingTraditional()
    b.set_data(aggregated)
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import MultiTaskLasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import LinearSVR
    from sklearn.svm import NuSVR
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import StackingRegressor
    a=MultiOutputRegressor(GradientBoostingRegressor())
    be=LinearRegression()
    c=MultiOutputRegressor(NuSVR())
    stacking_est=[("a",a),("b",be),("c",c)]


    rf =GradientBoostingRegressor(loss="ls") #LinearSVR(epsilon=.01,max_iter=4000) #MLPRegressor(alpha=.4,hidden_layer_sizes=(10,1))#LinearSVR(max_iter=10000) #KNeighborsRegressor(12,weights='distance') #RandomForestRegressor(max_depth=10) # RandomForestRegressor(max_depth=5,random_state=0)
    #rf=
    multra = MultiOutputRegressor(rf)
    b.set_model(multra)
    b.set_features(x_names)#,y_names) #y_names)
    print(b.score_cv())
    
    traditional_feature_prediction.save(b,"temp2.pkl")

    model2=traditional_feature_prediction.load("temp2.pkl")
    
    print(b.predict(b._x,preprocess=True)[0])
    print(model2.predict(b._x,preprocess=True)[0])
    print("OK")
    
