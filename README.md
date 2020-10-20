# ComicTransfer
A pipeline for quick positioning and font realignment for Japanese Mangas to English. This additionally has 
functionality for translation.

![](https://github.com/wumpusman/ComicTransfer/blob/experimental/gif_project.gif)
## Motivation
The process of editing mangas between languages is a cumbersome process 
that takes hours, and is primarily done manually.
Specifically this project focuses on automatically realigning text and positioning,
with inclusion, but with limited functionality for optional cleaning, and translation.

## Prerequisites
Currently, for core inference (core/main.py and app.py), this project has several requirements that are google cloud API dependent, and is expected to
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
<br><br>

## Running

### Run Streamlit App
**See prerequisites**

    streamlit run app.py
    optional: Docker build
    docker build -t mangnify-streamlit:v1 -f dockerfile.app .
    docker run -p 8501:8501 mangnfiy-streamlit:v1

### Run via main
**See prerequisites**

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

