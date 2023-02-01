FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

# Install Ubuntu Dependencies:
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --fix-missing --no-install-recommends apt-utils \
        build-essential \
        curl \
        binutils \
        gdb \
        git \
        freeglut3 \
        freeglut3-dev \
        libxi-dev \
        libxmu-dev \
        gfortran \
        pkg-config \
        libboost-python-dev \
        libboost-thread-dev \
        pbzip2 \
        rsync \
        software-properties-common \
        libboost-all-dev \
        libopenblas-dev \ 
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libgraphicsmagick1-dev \
        libswresample-dev \
        libavformat-dev \
        libhdf5-dev \
        libpq-dev \
        libgraphicsmagick1-dev \
        libavcodec-dev \
        libgtk2.0-dev \
        libgtk-3-dev\
        libv4l-dev\ 
        libxvidcore-dev\ 
        libx264-dev\
        libatlas-base-dev\
        liblapack-dev \
        liblapacke-dev \
        libswscale-dev \
        libcanberra-gtk-module \
        libboost-dev \
    	libboost-all-dev \
        libeigen3-dev \
	    wget \
        vim \
        qtbase5-dev \ 
        qtchooser \ 
        qt5-qmake \ 
        qtbase5-dev-tools \
        unzip \
	    zip \ 
        && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*  && \
        apt-get clean && rm -rf /tmp/* /var/tmp/*

ENV DEBIAN_FRONTEND noninteractive

# Install cmake version that supports anaconda python path
RUN wget -O cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v3.15.4/cmake-3.15.4-Linux-x86_64.tar.gz
RUN tar -xvf cmake.tar.gz
WORKDIR /cmake-3.15.4-Linux-x86_64
RUN cp -r bin /usr/
RUN cp -r share /usr/
RUN cp -r doc /usr/share/
RUN cp -r man /usr/share/
WORKDIR /
RUN rm -rf cmake-3.15.4-Linux-x86_64
RUN rm -rf cmake.tar.gz

# Copy usr/local Directory
RUN file="$(ls -1 /usr/local/)" && echo $file

ARG PYTHON=python3
ARG PIP=pip3

# Install Python:
RUN apt-get update && apt-get install -y ${PYTHON}.9

# Fix conda errors per Anaconda team until they can fix
RUN mkdir ~/.conda

# Install Anaconda(Miniconda)
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
/bin/bash Miniconda3-latest-Linux-x86_64.sh -f -b -p /opt/conda && \
rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/conda/bin:$PATH

# Install the correct version of gcc for opencv
RUN conda install -c conda-forge gcc=12.1.0

# Install pip for base conda env
RUN conda install pip
RUN conda install python=3.9
# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8


# Install Jupyter-lab:
RUN ${PIP} install jupyterlab

RUN ${PIP} install  mlflow \
                    ipykernel\
                    ipywidgets

# Some Anaconda envirnoment setup packages:
RUN conda update -n base -c defaults conda
RUN conda install pyg -c pyg
RUN conda update conda
RUN conda install numba
RUN conda install -c conda-forge protobuf
# RUN conda install captum -c pytorch

# Install OpenCV with GPU support built from source:
WORKDIR /
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN mv opencv-master opencv
RUN mv opencv_contrib-master opencv_contrib
RUN mkdir /opencv/build
WORKDIR /opencv/build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \ 
            -D BUILD_TIFF=ON \
		    -D BUILD_opencv_java=OFF \
		    -D WITH_CUDA=ON \
            -D CUDA_ARCH_BIN=7.5 \
		    -D ENABLE_FAST_MATH=1 \
		    -D CUDA_FAST_MATH=1 \
		    -D WITH_CUBLAS=1 \
		    -D ENABLE_AVX=ON \
		    -D WITH_OPENGL=ON \
		    -D WITH_OPENCL=OFF \
		    -D WITH_IPP=ON \
		    -D WITH_TBB=ON \
		    -D WITH_EIGEN=ON \
		    -D WITH_V4L=ON \
		    -D CMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
		    -D PYTHON3_EXECUTABLE=$(which python3) \
            -D PYTHON_DEFAULT_EXECUTABLE=${PYTHON3_EXECUTABLE} \
            -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
            -D PYTHON_INCLUDE_DIR2=$(python3 -c "from os.path import dirname; from distutils.sysconfig import get_config_h_filename; print(dirname(get_config_h_filename()))") \
            -D PYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_config_var;from os.path import dirname,join ; print(join(dirname(get_config_var('LIBPC')),get_config_var('LDLIBRARY')))") \
            -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
            -D OPENCV_ENABLE_NONFREE=ON \
            -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
            -D BUILD_EXAMPLES=ON \
            -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.1 \
            -D WITH_QT=ON ..

RUN make -j6 \
        && make install \
	    && rm /opencv.zip \
        && rm /opencv_contrib.zip \
    	&& rm -rf /opencv \
        && rm -rf /opencv_contrib

# Install Stable version of Numpy
RUN ${PIP} install --upgrade numpy --ignore-installed numpy

# pip install upgraded versions of these basic python packages always:
RUN ${PIP} install --upgrade-strategy only-if-needed pip \
                                                     setuptools \
                                                     hdf5storage \
                                                     h5py \
                                                     py3nvml \
                                                     pyinstrument\
                                                     scikit-image \
                                                     imgaug \
                                                     scikit-learn \
                                                     pydot\
                                                     matplotlib \
                                                     pandas \
                                                     seaborn \
                                                     scipy \
                                                     yellowbrick \
                                                     plotly\ 
                                                     --ignore-installed numpy


# Install pytorch the latest stable version
RUN ${PIP} install torch torchvision torchaudio 

# Install specific version of tensorflow
RUN ${PIP} install tensorflow

WORKDIR /app
EXPOSE 8888 6006

# Better container security versus running as root
RUN useradd -ms /bin/bash container_user

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab  --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.custom_display_url='http://localhost:8888' "]

