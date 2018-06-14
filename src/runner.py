"""
Author: Rainer Killinger rainer@killinger.co

This script generates a parameter grid for the char_lstm
hyperparamters, runs each of them and saves result for a tag to evaulate
"""

import itertools
import os
import glob
import re
import sys
import numpy as np
from collections import defaultdict
import subprocess as sub
import operator
from keras.models import Sequential, load_model, Model
from pathlib import Path
import operator
import pickle
from time import time
import datetime 
import pytz

compare_to_cat = True
uses_cat = [False,True] if compare_to_cat else [True]

iob_file = "iob-files/en_sentences.iob"
tags_to_evaluate = ["S"]
modelzoo_prefix = "en_s/en_s"
sequence_lengths = [7,20,40,90,120]
lstm_units = [32,64,128]
dropouts = [0.0,0.2,0.4]

#iob_file = "iob-files/de_clauses_and_sentences.iob"
#tags_to_evaluate = ["C"]
#modelzoo_prefix = "de_c_s/de_c_s"
#sequence_lengths = [20,40]
#lstm_units = [32,64,128]
#dropouts = [0.0,0.2,0.4]

def expand_parameter(parameters):
    result = defaultdict(list)
    repeatlength= len(parameters) 
    product = list(itertools.product(*list(parameters)))
    for i,parametertuple in enumerate(product):
        for j,parameter in enumerate(parametertuple):
            result.setdefault(j, []).append(parameter)
    return tuple(result.values())

sequence_lengths,uses_cat,lstm_units,dropouts = expand_parameter((sequence_lengths,uses_cat,lstm_units,dropouts))
print("Running training with input from '{}'".format(iob_file))
start = None
end = None
for index,parameter in enumerate(sequence_lengths):
    use_cat = "--cat" if uses_cat[index] else ""
    command = "python3 lstm_training.py -p {} -l {} -u {} -d {} {} -i {}".format(modelzoo_prefix,
                                                                                str(sequence_lengths[index]),
                                                                                str(lstm_units[index]),
                                                                                str(dropouts[index]),
                                                                                use_cat,iob_file)
    eta = "none"
    if end is not None and end - start > 5.0:
        eta_date = datetime.datetime.now(pytz.utc) + datetime.timedelta(seconds=(end - start)*(len(sequence_lengths)-index))
        eta = str(eta_date.replace(microsecond=0)) + " UTC"
    print("ETA {} - model: {} / {} - parameters: sequence_lengths {} lstm_units {} dropout {} uses cat {}".format(eta,
                                                                                                                                str(index+1),
                                                                                                                                str(len(sequence_lengths)),
                                                                                                                                str(sequence_lengths[index]),
                                                                                                                                str(lstm_units[index]),
                                                                                                                                str(dropouts[index]),
                                                                                                                                str(uses_cat[index])), end="\r")
    start = time()
    process = sub.call(command.split(),stdout=open(os.devnull, "wb"),stderr=open(os.devnull, "wb"))
    end = time()
    print(" " * 150, end="\r" ) # clear line

if len(tags_to_evaluate) == 0 or tags_to_evaluate is None:
    sys.exit(0)

if len(list(glob.iglob("modelzoo/"+modelzoo_prefix+"_*"))) == 0 :
    sys.exit(0)

evaluation_elements = {}
model_history = sorted(glob.glob("modelzoo/"+modelzoo_prefix+"_*/checkpoints/evaluation.pickle"), key=os.path.getmtime, reverse=True)

for historypath in model_history:
    path = Path(historypath)
    modelname = path.parts[-3]
    with open(historypath, "rb") as eval_file:
        model_evaluation= pickle.load(eval_file)
    cm = model_evaluation["cm"][-1].astype("float") 
    for tag in tags_to_evaluate:
        tag_index = model_evaluation["tags"].index(tag)
        np_cm = np.array(cm)
        tP = np_cm[tag_index,tag_index]
        fP = np.sum(np_cm[:,tag_index])-tP
        fN = np.sum(np_cm[tag_index,:])-tP
        evaluation_elements[modelname] = (2*tP) / ((2*tP) + fP + fN)

evaluation_sorted = sorted(evaluation_elements.items(), key=operator.itemgetter(1), reverse=True)

result = ["modelname,eval"]
for key,value in evaluation_sorted:
    result.append("{},{}".format(key,value))
with open("modelzoo/"+modelzoo_prefix+"_result.csv", "w") as text_file:
    text_file.write("\n".join(result))
