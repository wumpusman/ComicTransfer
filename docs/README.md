## Directory Outline

    |-- core 
    |   |-- components
    |   |   |-- alignment
    |   |   |-- assignment
    |   |   |-- clean
    |   |   |-- translation
    |   |-- datahandling
    |   |-- models
    |   |   |-- deepnn
    |   |-- scraping
    |   |-- training
    |       |-- deepnn
    |-- data
    |-- docs
    |-- tests
        |-- core
            |-- components
            |-- datahandling
            |-- models

## Files And Directories
    app.py: Primary streamlit app interface - streamlit run app.py
    
    data/: A folder containing temporary data files, fonts and models for tests
    data/craft_text: a folder that includes sample training data and outputs for more experimental deepnn models
    data/models/: a folder that includes sample models for font and bounding predictions
    
    core/: primary location of functionality for training, models, inference, scraping, etc.
    core/main.py: contains method for running image processing and analysis independent of the app
     
    core/components: components pertaining to full pipeline of inference (box detection, cleaning, ocr, translation, reassignment)
    core/components/pipeline.py: a class for handling processing of image and text data and outputs various results for inference
    
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
    core/datahandling/datahandler_img.py: script that contains functions and classes for preparing images for cnn architecture
    
    core/models/: primary location for any models that are used for predicting or inference, these are incorporated into the objects of /core/components/
    core/models/traditional_feature_prediction.py: contains base class for predicting font size and bounding box size
    core/models/iou_baseline.py: contains class for evaluating naive baseline of predicting bounding box
    core/models/iou_prediction.py: contains class for evaluating bounding box using variable info to predict x1,y1,x2,y2 (top left, bottom right coords)
    core/models/iou_extended.py: contains class for evaulating bounding but predicts width and height separate from raw coords
    
    core/models/deepnn/: scripts for handling cnn architectures and anything other NN architectures
    core/models/deepnn/coordconv.py: modification of a preexisting implementation of coordconv - effectively adds position information to channels
    core/models/deepnn/craft_wtih_coord.py: contains class for nn architecture that does a prelim pass at adding a coordconv to output of textcraft models
    
    core/scraping/: scripts for extracting images, website html info and meta info
    core/scraping/main_extraction.py: script for running selenium extraction of images and associated meta files
    core/scraping/extract_img_selenium.py: code that extracts image pairs and extranous info from selenium parsed page
    core/scraping/extract_info.py: code that extracts raw info from html of the page

    core/training/: scripts for handling training or evaluation of various models - a place of where dreams go to die, and realism is born
    core/training/train_feature_ablation.py: script for training font size or bounding box using largely feature engineering
    core/training/train_skauto_ml.py: (deprecated) script for evaluating skautoml, hyperparameter, package had instabilities, leaving for posterity
    core/training/eval_automl_cnn.py: (online model disabled) a script that requires an accessible pretrained model setup in google automl package, used for getting prelim results on bounding box transfer
    
    core/training/deepnn/: scripts for training  the cnn nn models for detection and prediction of bounding boxes (currently transitioning from earlier notebooks) 
    core/training/deepnn/train_craftcoord.py: training with a textcraft model with additional coordconv network on top
    core/training/deepnn/tune_textcraft.py: meant for simply tuning a textcraft model on new data with bounding boxes
    
    tests/: contains primary code for evaluating the current core functionality used in the app
    tests/core/datahandling/test_process_bilingual_data.py: evaluates basic tests that features processed are correct
    tests/core/models/test_iou_baseline.py: scripts for testing baseline functionality
    tests/core/models/test_iou_prediction.py: scripts for bounding training and prediction
    tests/core/models/test_traditional_feature_prediction.py: scripts for training and prediction of font size
    
    