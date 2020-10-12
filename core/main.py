import argparse
import cv2
from PIL import Image, ImageFont, ImageDraw 
import sys
import os
sys.path.append("../")
from core.components.alignment import predict_jp_bounding
from core.components.translation import predictor_translate
from core.components.clean import clean_img
from core.components.assignment import assign_text
from core.components.assignment import assign_ml
from core.components import pipeline



default_image_path:str="../data/temp.png"
default_font_model_pth="../data/models/font_model.pkl"
default_bound_model_pth="../data/models/bounding_model.pkl"
default_font_pth='../data/LiberationMono-Bold.ttf'
default_save_dir="../data/results_temp"
default_project_id="typegan" #the one i chose, but you should assign your relevant one

parser = argparse.ArgumentParser(description='code for running extraction pipeline for a local image')
parser.add_argument("-p","--file_path",help="path to a jpg or png",default=default_image_path)
parser.add_argument("-f","--font_path",help="path to font to be used",default=default_font_pth)
parser.add_argument("-s","--model_font_size",help="path to model for font size",default=default_font_model_pth)
parser.add_argument("-b","--model_bounding_box",help="path to model for bounding box",default=default_bound_model_pth)
parser.add_argument("-i","--project_id",help="project id associated with a project that has permission for google cloud services",default=default_project_id)
parser.add_argument("-o","--output",help="where are you saving the results",default=default_save_dir)


def main(project_id:str,image_path:str,destination:str,font_path:str,font_model_path:str,bound_model_path:str):
    """
    runs through the pipeline to extract an image with access to various features embedded in the pipeline_obj
    Args:
        project_id: project id associated with the google project, this entitles you to particular permissions for online deployment
        image_path: path to image you wish to process
        destination: path to save a sample of the results
        font_path: path to a .tff font that is used to draw images
        font_model_path: model for estimating font size
        bound_model_path: model for estimating how bounds of text should be in english

    Returns:
        None
    """

    assign_obj = assign_ml.load_default_model(font_model_path, bound_model_path)
    pipeline_obj = pipeline.PipeComponents(project_id,font_path)
    pipeline_obj.set_assignment_model(assign_obj)
    pipeline_obj.calculate_results_from_path(image_path)

    print(pipeline_obj._image_overlaid_text)

    image_to_draw= Image.fromarray(pipeline_obj._image_overlaid_text)


    image_to_draw.save(os.path.join(destination,"img_overlay2.png"))

if __name__ == '__main__':
    args = parser.parse_args()

    main(args.project_id, args.file_path, args.output, args.font_path, args.model_font_size, args.model_bounding_box)

    
    
    
    
    
    