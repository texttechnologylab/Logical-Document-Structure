#keras version 2.1.4
FROM nvidia/cuda:9.0-cudnn7-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget git libhdf5-dev g++ graphviz openmpi-bin libgl1-mesa-glx && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

USER keras

# Python
ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    pip install --upgrade pip && \
    pip install -Iv tensorflow-gpu==1.5.0 && \
    pip install https://cntk.ai/PythonWheel/GPU/cntk-2.1-cp36-cp36m-linux_x86_64.whl && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    conda install theano pygpu bcolz && \
    pip install sklearn_pandas && \
    git clone git://github.com/keras-team/keras.git /src && cd src && git reset --hard 5d54eeb3967f3364955478c9520ed9ba05c4f9f2 && cd .. && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git@5d54eeb3967f3364955478c9520ed9ba05c4f9f2 && \
    pip install pydot3 && \
    pip install beautifulsoup4 && \
    pip install pathlib && \
    pip install statsmodels && \
    pip install seaborn && \
    conda clean -yt

ADD theanorc /home/keras/.theanorc

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /src/workdir
ADD src /src/workdir

EXPOSE 8080

CMD jupyter notebook --port=8080 --ip=0.0.0.0
