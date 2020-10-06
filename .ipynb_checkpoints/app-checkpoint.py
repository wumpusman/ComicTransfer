import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import os
import io
import PIL.Image as Image
from core.components.alignment import predict_jp_bounding
from core.components.translation import predictor_translate
from core.training.feature_engineering.traditional_feature_prediction import FeaturePredictionTraditional
from core.training.feature_engineering.iou_prediction import PredictionBoundingTraditional
from core.components.clean import clean_img
from core.components.assignment import assign_text
from core.components.assignment import assign_ml
from core.components import pipeline
from streamlit.elements import image_proto

from google.cloud import vision
import io

image_proto.MAXIMUM_CONTENT_WIDTH = 600
image_proto.MAXIMUM_CONTENT_HEIGHT= 600
st.set_option('deprecation.showfileUploaderEncoding', False)
data_name = "Yaiba_selenium.tsv" #Doraemon_Long_Stories_selenium.tsv"

full_path = data_name #os.path.join(data_path, data_name)
#st.write(os.listdir())
#st.write(full_path)
all_manga = pd.read_csv(full_path, sep="\t", index_col=0)
#st.write(len(all_manga))
all_manga = all_manga.drop(columns=["level_0"])

app_mode = st.sidebar.selectbox("Choose the model",
        ["Naive", "Model-Ensemble"])


st.title("Manga Translate")
st.markdown("------KILL ME-----")
uploaded_file=None

pipeline=pipeline.PipeComponents()
pipeline.set_assignment_model(assign_ml.load_default_model())

st.write("FUUUU")
uploaded_file = st.file_uploader("")

bounder_google=predict_jp_bounding.BoundingGoogle(True)
text_translation=predictor_translate.TranslationGoogle("typegan")
cleaning_obj=clean_img.CleanDefault()


text_image_assignment=assign_text.AssignDefault() #assign_ml.load_default_model()
text_image_assignment2=assign_ml.load_default_model()



if type(uploaded_file)!=type(None):
    
    
    client = vision.ImageAnnotatorClient()

    #with io.open(path, 'rb') as image_file:
    #content = uploaded_file.read()
  
    
    results=bounder_google.predict_byte(uploaded_file)
    results_structured=bounder_google.format_predictions(results)
    text_estimates=text_translation.predict(results["text"])
    
    display_trans=pd.DataFrame()
    display_trans["jp"]=results["text"]
    display_trans["pr"]=text_estimates
    
    #display_trans

    image = Image.open(uploaded_file)
    st.image(image)
    #st.write(type(image))
    #st.write(results)
    numpy_version=np.copy(np.asarray(image))
    
    cleaned_image=cleaning_obj.clean(numpy_version,results["vertices"])
    
    
    
    #st.image(Image.fromarray(cleaned_image))
    
    
    font_path='/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'
    result_image=None
    if app_mode =="Naive":
        result_image=text_image_assignment.assign_all(cleaned_image,text_estimates,results_structured,font_path)
    if app_mode =="Model-Ensemble":
        result_image=text_image_assignment2.assign_all(cleaned_image,text_estimates,results_structured,font_path)
    st.image(Image.fromarray(result_image))
    image_mod=np.copy(np.asarray(numpy_version))

   
    image_mod[0:200,0:200]=0
    #image= Image.fromarray(image_mod.T)
   
    image=Image.fromarray(image_mod)
    
    
    
#dict_lang={"Japanese":"jp"}
#dict_lang.update({"English":"en"})


#st.write(app_mode)

#language=all_manga[all_manga["language"]==dict_lang[app_mode]].text
#language=language.reset_index(drop=True)
#language