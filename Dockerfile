FROM ubuntu:20.04
	
# 设置环境变量，允许 root 用户运行 MPI
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ARG DEBIAN_FRONTEND=noninteractive
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# 安装所有依赖（一次性安装，减少层数）
RUN apt-get update -y && \
    apt-get install -y wget curl git sudo
    
#=============================================================================================
#  Set up Python Jupyter Environment ...
#=============================================================================================

ARG url0=https://github.com/conda-forge/miniforge/releases/download/22.9.0-2/Miniforge3-22.9.0-2-Linux-x86_64.sh
ARG url0=https://github.com/conda-forge/miniforge/releases/download/4.12.0-0/Miniforge3-4.12.0-0-Linux-x86_64.sh

RUN wget --quiet ${url0} -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/miniconda3 \
    && rm ~/miniconda.sh \
    && ln -s /opt/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH=/opt/miniconda3/bin:${PATH}

RUN . /root/.bashrc \
    && /opt/miniconda3/bin/conda init bash \
    && conda info --envs
	
# 创建CONDA环境来安装DL4DS降尺度软件

ARG DL4DS=true

RUN if [ "$DL4DS" = true ]; then \
    echo "install DL4DS ..."; \
    . /root/.bashrc; \ 
    conda create -n dl4ds_py39_cu11 -c conda-forge python==3.9.* xarray cartopy cudatoolkit==11.* cudnn==8.* numpy==1.* -y; \
    conda activate dl4ds_py39_cu11; \
    which python ;\
#    pip install tensorflow==2.10.* climetlab climetlab_maelstrom_downscaling numpy==1.* ; \
    pip install git+https://github.com/wk1984/dl4ds_fixed.git ; \
    python -c "import tensorflow as tf; print('Built with CUDA:', tf.test.is_built_with_cuda(), 'USE GPU:', tf.config.list_physical_devices('GPU'))"; \
 	python -c "import dl4ds as dds"; \
	fi
	    
RUN useradd -m -s /bin/bash user && echo "user:111" | chpasswd
RUN usermod -aG sudo user

USER user
WORKDIR /work

RUN python -c "import dl4ds as dds"