# It is important that cuda supports the video card architectures that are important to us:
# NVIDIA GeForce RTX 3060 / RTX 3060 Ti - sm_86
# NVIDIA GeForce RTX 2080 Ti            - sm_75
# NVIDIA A100-SXM4-40GB                 - sm_80
# docker build -t quay.io/logivations/ml_all:LS_dfine_latest .

ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx 
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

COPY . /DEIM
WORKDIR /DEIM

RUN pip install -r requirements.txt