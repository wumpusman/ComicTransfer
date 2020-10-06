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
from core.components import pipeline
from core.components.assignment import assign_ml


@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_pipeline_model(note):
    pipeline_obj = pipeline.PipeComponents()
    pipeline_obj.set_assignment_model(assign_ml.load_default_model())
    return pipeline_obj

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def get_ml_model():
    return assign_ml.load_default_model()

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def get_default_model():
    return assign_text.AssignDefault()

pipeline_obj =load_pipeline_model("")
st.title("Japanese To English Manga Annotation")

image_proto.MAXIMUM_CONTENT_WIDTH = 700
image_proto.MAXIMUM_CONTENT_HEIGHT= 600
st.set_option('deprecation.showfileUploaderEncoding', False)



model_mode = st.sidebar.selectbox("Choose the model",
        ["Model-Ensemble","Base"])


st.sidebar.write("---------")
st.sidebar.write("**      Upload Japanese Comic Page**")
uploaded_file=st.sidebar.file_uploader("")
click_begin=False

if type(uploaded_file)!=type(None):
    click_begin=st.sidebar.button("Begin")

if click_begin:
    pipeline_obj.clear_prev_estimates()
    if model_mode=="Model-Ensemble":
        pipeline_obj.set_assignment_model(get_ml_model())
    else:
        pipeline_obj.set_assignment_model(get_default_model())





data_to_present_names= ["Raw Image","Quick Clean","Mask","Overlay Text","Annotated Information"]
data_to_present:str=st.selectbox('', data_to_present_names)
st.write("----------------")

if click_begin or pipeline_obj.has_run():

    if (pipeline_obj.has_run()==False):

        pipeline_obj.calculate_results(uploaded_file)


    if data_to_present ==data_to_present_names[0]:
        st.image(Image.fromarray(pipeline_obj._image_unprocessed))
    if data_to_present==data_to_present_names[1]:
        st.image(Image.fromarray(pipeline_obj._image_cleaned))
    if data_to_present==data_to_present_names[2]:
        st.image(Image.fromarray(pipeline_obj._image_text_mask))
    if data_to_present==data_to_present_names[3]:
        st.image(Image.fromarray(pipeline_obj._image_overlaid_text))
    if data_to_present == data_to_present_names[4]:
        st.write(pipeline_obj._data_estimates)


    
    
