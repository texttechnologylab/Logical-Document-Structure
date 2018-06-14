# Logical-Document-Structure

Logical Document Structure Segmentation via Keras character-based LSTM Architecture.
(Erkennung logischer Dokumentstukturen und Satzstrukturen mit Hilfe von neuronalen Netzen)

## General Information

Setup is easy if you get [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) Version 1 or 2 up and running.

Otherwise you'll have to install:

* Python 3
* Nvidia Drivers and CUDA 9.0 with cuDNN 7.0
* Keras Version 2.1.4
* Python modules: Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py theano pygpu bcolz pydot3 beautifulsoup4 pathlib statsmodels seaborn 

## Usage

Training is based on a csv formatted file consisting of unicode category, unicode codepoint and IOB-Token (you can choose your own tokens) for each character (row-wise).
For an example segmenting english sentences and tokens based on the [Groningen Meaning Bank (GMB)](http://gmb.let.rug.nl/data.php) you can take a look at the *keras\_src/iob-files/en_sentences.iob* file.

To start a training run use **lstm_training.py** with custom parameters like number of LSTM units, dropout, IOB file path etc. .
You can call the important scripts with -h to get help.
All output of a training run will land in the *modelzoo* directory.
To configure and run a training via a parameter grid use (just change for docker) **runner.py**.
To get an overview of the performance of the trained models via **runner.py** it will generate an csv formatted file containing metrics that can be visualised with **swarmplot_scores.py**.
The *modelzoo* directory contains examples (only one model was committed to this repo).

After finishing training the script called **segmenter.py** can be used to segment stdin or a file given a trained model.


## Docker

This directory contains a modified Dockerfile from the [Keras project](https://github.com/keras-team/keras/commits/master/docker/Dockerfile).
To get a interactive shell (Bash) with all the dependencies resolved in a docker container use make:

```bash
make bash 
```

The **runner.py** script can also be called using docker. 

```shell
make run 
```

### LSTMVIS

If you're familiar with [LSTMVis](https://github.com/HendrikStrobelt/LSTMVis) you can use **segmenter.py** and the scripts in *src/lstmvis* to reformat training data and output for LSTMVis. Examples are included in the same directory.
