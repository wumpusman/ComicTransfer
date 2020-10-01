import argparse
import cv2
from PIL import Image, ImageFont, ImageDraw 
import sys
sys.path.append("../")
from core.components.alignment import predictor_bounding
from core.components.translation import predictor_translate
from core.components.clean import clean_img
from core.components.assignment import assign_text

default_image_path:str="/home/jupyter/ComicTransfer/data/temp.png"
parser = argparse.ArgumentParser(description='sample for running the converter with default')
parser.add_argument("-p","--file_path",help="path to a jpg or png",default=default_image_path)
parser.add_argument("-f","--font_type",help="path to font to be used",default='/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf')
parser.add_argument("-d","--destination",help="path to destination folder",default="temp_output.jpg")




def main(image_path:str,destination:str,font_path:str):
    """
    
    """
    bounder_google=predictor_bounding.BoundingGoogle(True) 
    text_translation=predictor_translate.TranslationGoogle("typegan")
    text_image_assignment=assign_text.AssignDefault()
    cleaning_obj=clean_img.CleanDefault()

    image_cv=cv2.imread(image_path,cv2.IMREAD_COLOR)
    
    results=bounder_google.predict(image_path)
    text_estimates=text_translation.predict(results["text"])
    
    cleaned_image=cleaning_obj.clean(image_cv,results["vertices"])
    result_image=text_image_assignment.assign_all(cleaned_image,text_estimates,results["vertices"],font_path)
    image_to_draw= Image.fromarray(result_image)
    image_to_draw.save(destination)

if __name__ == '__main__':
    args = parser.parse_args()
    
    main(args.file_path,args.destination,args.font_type)
    
    
    
    
    
    