FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN wget https://raw.githubusercontent.com/GrebenkovaO/MinkowskiFCD/main/requirements.txt
RUN apt update && apt install python3-pip -y
RUN apt install build-essential python3-dev libopenblas-dev git vim -y
RUN apt update --fix-missing && apt install ffmpeg libsm6 libxext6  -y

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

RUN git clone https://github.com/NVIDIA/MinkowskiEngine && cd MinkowskiEngine && \
    ls ./ && python3 setup.py install && cd /

