FROM python:3.10.10-slim

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install --no-install-recommends -y build-essential=12.9 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y  && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir \
    cmake \
    numpy \ 
    scikit_image \
    opencv_python \
    Pillow \
    imutils
    #skimage \
RUN pip install dlib 
COPY . /face_morph
