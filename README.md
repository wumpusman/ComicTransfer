# ComicTransfer
A pipeline for quick positioning and font realignment for Japanese Mangas to English. This additionally has 
functionality for translation.

## Motivation
The process of editing mangas between languages is a cumbersome process 
that takes hours, and is primarily done manually.
Specifically this project focuses on automatically realigning text and positioning,
with inclusion, but with limited functionality for optional cleaning, and translation.

## Prerequisites
Currently, for core inference (/core/main.py and app.py), this project has several requirements that are google cloud API dependent, and is expected to
be embedded in a google compute instance. This requirements include access to google vision api, and google translate api. 
For setup: [UI setup](https://cloud.google.com/compute/docs/quickstart-linux) , 
[Interfacing](https://cloud.google.com/compute/docs/ssh-in-browser)
<br><br>
Please note, there is functionality for scraping data, testing and  training outside of those
dependencies


## Setup
    *optional: install conda
    *optional: conda create --name ComicTransfer python=3.7
    *optional: conda activate ComicTransfer
    clone your repository into your vm instance

    Run:
    
    git clone https://github.com/wumpusman/ComicTransfer
    cd ./ComicTransfer
    pip install -r requirements.txt
<br>

## Running

### Run Streamlit App
    streamlit run app.py
    optional: Docker build
    docker build -t mangnify-streamlit:v1 -f dockerfile.app .
    docker run -p 8501:8501 mangnfiy-streamlit:v1

### Run via main

    cd /core
    python main.py 
    
    example usage:
    '''
    python main.py -p ../data/temp.png -o ../data/results_temp/ --model_font_size ../data/models/font_model.pkl
    '''
    
### Retraining Models
    cd /core
    python training/train_feature_ablation.py 
    
    example usage:
    '''
    python  training/train_feature_ablation.py -d ../data/bilingual_tsv --savepath  ../data/models_temp/bounding_model2.pkl --savemodel True --type font
    '''

### Recollecting Data
ensure chromedriver is installed and referenced [ubuntu setup](https://www.srcmake.com/home/selenium-python-chromedriver-ubuntu)
    
    cd /core
    python scraping/main_extraction.py
    
    example usage:
    '''
    python scraping/main_extraction.py -m "../data/manga_list.txt" --number_per_manga 3
    '''

   
### Tests
    pytest tests
    
    example usage:
    '''
    pytest tests/core/datahandling/test_process_bilingual_data.py
    ''' 

## Files And Directories
    app.py: Primary streamlit app interface - streamlit run app.py
    
    data/: A folder containing temporary data files, fonts and models for tests
    data/results_temp/: A folder for sample results such as image outputs
    data/models/: A folder that includes sample models for font and bounding predictions
    
    core/: primary location of functionality for training, models, inference, scraping, etc.
    core/main.py: contains method for running image processing and analysis independent of the app
     
    core/components: Components pertaining to full pipeline of inference (box detection, cleaning, ocr, translation, reassignment)
    core/components/pipeline.py: A class for handling processing of image and text data and outputs various results for inference
    
    core/components/alignment: objects for extracting box area, intended to be expandable to handle different models
    core/components/alignment/predict_jp_bounding.py: Contains classes for extracting the japanese bonuding box, and ocr, currently uses google vision api
     
    core/components/assignment: objects for handling font size estimation for english, and how bounding box should be repositioned - meant to be extendable
    core/components/assignment/assign_text.py: contains class for naively estimating bounding info and font size
    core/components/assignment/assign_ml.py: contains class for incorporating models for predicting font size and bounding box information
    
    core/components/clean: objects for handling cleaning the original text and refilling
    core/components/clean/clean_img.py: contains classic for rudimentary cleaning based purely on original bounding box
       
    core/components/translation: objects for handling translation
    core/components/translation/predictor_translate.py: simple class that extracts translations using google translate api
    
    core/datahanding/: primary location for parsing or preparing data for downstream objects and training or inference
    core/datahanding/process_bilingual_data.py: contains an object that handles parsing data scraped by bilingual manga and creates additional features, used for preparing data for inference and training
    core/datahandling/process_folders_for_automl.py: script that prepares downloaded data into a single csv and path to be used in Google Cloud automl CNN 
    
    core/models/: primary location for any models that are used for predicting or inference, these are incorporated into the objects of /core/components/
    core/models/traditional_feature_prediction.py: contains base class for predicting font size and bounding box size
    core/models/iou_baseline.py: contains class for evaluating naive baseline of predicting bounding box
    core/models/iou_prediction.py: contains class for evaluating bounding box using variable info to predict x1,y1,x2,y2 (top left, bottom right coords)
    core/models/iou_extended.py: contains class for evaulating bounding but predicts width and height separate from raw coords
    
    core/scraping/: scripts for extracting images, website html info and meta info
    core/scraping/main_extraction.py: script for running selenium extraction of images and associated meta files
    core/scraping/extract_img_selenium.py: code that extracts image pairs and extranous info from selenium parsed page
    core/scraping/extract_info.py: code that extracts raw info from html of the page

    core/training/: scripts for handling training or evaluation of various models - a place of where dreams go to die, and realism is born
    core/training/train_feature_ablation.py: script for training font size or bounding box using largely feature engineering
    core/training/train_skauto_ml.py: (deprecated) script for evaluating skautoml, hyperparameter, package had instabilities, leaving for posterity
    core/training/eval_automl_cnn.py: (online model disabled) a script that requires an accessible pretrained model setup in google automl package, used for getting prelim results on bounding box transfer
    
    tests/: contains primary code for evaluating the current core functionality used in the app
    tests/core/datahandling/test_process_bilingual_data.py: evaluates basic tests that features processed are correct
    tests/core/models/test_iou_baseline.py: scripts for testing baseline functionality
    tests/core/models/test_iou_prediction.py: scripts for bounding training and prediction
    tests/core/models/test_traditional_feature_prediction.py: scripts for training and prediction of font size
    
    