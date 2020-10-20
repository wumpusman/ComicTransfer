FROM python:3.7-slim
EXPOSE 8501
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
COPY app.py app.py
COPY data data
COPY core core
COPY requirements_docker.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip install pillow
CMD streamlit run app.py