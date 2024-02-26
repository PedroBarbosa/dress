
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp

RUN apt-get update && \
    apt install -y git wget
# libopenmpi-dev \
# ghostscript \
# libgs-dev \
# libgd-dev \
# libexpat1-dev \
# zlib1g-dev \
# libxml2-dev \
# autoconf automake libtool \
# libhtml-template-compiled-perl \
# libxml-opml-simplegen-perl \
# libxml-libxml-debugging-perl 

#build-essential zlib1g-dev autoconf automake libtool libexpat-dev libxml2-dev

# # Install python 3.11 in the system
# RUN add-apt-repository 'ppa:deadsnakes/ppa' 
# RUN apt update && apt install -y python3.11 && ln -sf /usr/bin/python3.11 /usr/bin/python && ln -sf /usr/bin/python3.11 /usr/bin/python3
# RUN apt install -y python3-pip

# # Install cuda v12 libraries
# RUN apt-get install -y linux-headers-5.15.0-57-generic
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb 
# RUN dpkg -i cuda-keyring_1.0-1_all.deb
# RUN apt-get update
# #RUN apt-get update && apt-get --no-install-recommends install -y cuda
# #ENV PATH="$PATH:/usr/local/cuda-12.0/bin"

# Mamba with pyton 3.10
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh && bash Mambaforge-Linux-x86_64.sh -b -p /opt/mamba 
ENV PATH="/opt/mamba/bin:${PATH}" 

#RUN mamba install -y -c nvidia/label/cuda-12.3.0 cuda
RUN mamba install -y -c conda-forge python=3.10.10 cudatoolkit cudnn gcc
#cudatoolkit=11.2.2 cudnn=8.1.0 gcc

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mamba/lib
RUN python -m pip install tensorflow==2.11.0 loguru jsonschema tqdm pandas==2.0.3 pandarallel biopython spliceai==1.3.1 pyranges scikit-learn seaborn statsmodels dna_features_viewer rich_click pyyaml
RUN mamba install -y -c bioconda meme==5.5.3

# MEME suite v5.5.3
# RUN wget https://meme-suite.org/meme/meme-software/5.5.3/meme-5.5.3.tar.gz && tar -xvzf meme-5.5.3.tar.gz && cd meme-5.5.3 && ./configure --prefix=/opt/meme --enable-build-libxml2 --enable-build-libxslt && make && make test && make install
# ENV PATH="/opt/meme/bin:/opt/meme/libexec/meme-5.5.3:${PATH}"

WORKDIR /work
RUN rm -rf /tmp

# GeneticEngine
RUN pip install --upgrade --no-cache-dir GeneticEngine==0.8.5

# Pangolin
RUN pip install torch --index-url https://download.pytorch.org/whl/cu113
RUN pip install git+https://github.com/tkzeng/Pangolin.git@5cf94b8

VOLUME /work
