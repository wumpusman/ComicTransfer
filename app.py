import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import os
import io
import PIL.Image as Image
from core.components.alignment import predictor_bounding
from core.components.translation import predictor_translate
from core.components.clean import clean_img
from core.components.assignment import assign_text
from streamlit.elements import image_proto

from google.cloud import vision
import io

image_proto.MAXIMUM_CONTENT_WIDTH = 600
image_proto.MAXIMUM_CONTENT_HEIGHT= 600
st.set_option('deprecation.showfileUploaderEncoding', False)
data_path = "C://Users//egasy//Downloads//ComicTransfer//ComicTransfer//ExtractBilingual/bi/"
data_name = "Yaiba_selenium.tsv" #Doraemon_Long_Stories_selenium.tsv"

full_path = data_name #os.path.join(data_path, data_name)
#st.write(os.listdir())
#st.write(full_path)
all_manga = pd.read_csv(full_path, sep="\t", index_col=0)
#st.write(len(all_manga))
all_manga = all_manga.drop(columns=["level_0"])

app_mode = st.sidebar.selectbox("Choose the data mode",
        ["Japanese", "English"])


st.title("Manga Translate")
st.markdown("-----------")
uploaded_file=None




uploaded_file = st.file_uploader("")

bounder_google=predictor_bounding.BoundingGoogle(True) 
text_translation=predictor_translate.TranslationGoogle("typegan")
cleaning_obj=clean_img.CleanDefault()
text_image_assignment=assign_text.AssignDefault()


if type(uploaded_file)!=type(None):
    
    
    client = vision.ImageAnnotatorClient()

    #with io.open(path, 'rb') as image_file:
    #content = uploaded_file.read()
  
    
    results=bounder_google.predict_byte(uploaded_file)
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
    
    
    
    st.image(Image.fromarray(cleaned_image))
    
    
    font_path='/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'
    result_image=text_image_assignment.assign_all(cleaned_image,text_estimates,results["vertices"],font_path)
    
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