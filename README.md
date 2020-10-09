# ComicTransfer
A pipeline for quick positioning and font realignment for Japanese Mangas to English

##Motivation
The process of editing mangas between languages is a cumbersome process 
that takes hours, and is primarily done manually.
Specifically this project addresses an issue , automatically realigning text and positioning,
with optional cleaning, and translation.

##Prerequistes
Currently, for core inference, this project has several requirements that are google cloud API dependent, and is expected to
be embedded in a google compute instance. 
<br>
For UI setup see: [UI setup] (https://www.waferwire.com/blog/creating-a-virtual-machine-instance-with-google-cloud/)


This requirements include access to google vision api, and google translate api. 
However, there is functionality for testing, scraping data, and inital training outside of those
dependencies (see app_local.py, tests/)


## Setup
Clone your repository into your vm instance
<br>
<i>Run</i>:
<br>
git clone https://github.com/wumpusman/ComicTransfer
<br>
cd ./ComicTransfer
<br>
conda create --name ComicTransfer python=3.7
<br>
conda activate ComicTransfer
<br>
pip install -r requirements.txt
<br>

##Run Streamlit App
streamlit run app.py
<br>
Optional: Docker build
<br>
docker build -t mangnify-streamlit:v1 -f dockerfile.app .
<br>
docker run -p 8501:8501 mangnfiy-streamlit:v1