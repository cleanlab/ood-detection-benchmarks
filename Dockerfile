FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04
WORKDIR /dcai
COPY . .
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && \
    apt-get -y install make python3 python3-pip ffmpeg libsm6 libxext6 git && \
    python3 -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
ENV PYTHONPATH=".:${PYTHONPATH}"