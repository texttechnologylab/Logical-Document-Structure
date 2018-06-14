# coding=utf-8
"""
Author: Rainer Killinger rainer@killinger.co
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from keras.models import Sequential, load_model
from keras.layers import concatenate, Lambda
from bs4 import BeautifulSoup, UnicodeDammit

import numpy as np
import random
import sys
import pandas as pd
import itertools
import math
import h5py

import glob
import re
import pickle
import argparse
import unicodedata


def split_at_indices(string, indices):
    consumed = 0
    list_of_strings = []
    for i,splitter in enumerate(indices):
        if splitter is 0:
            continue
        list_of_strings.append(string[consumed:splitter])
        consumed=splitter
        if i is len(indices)-1:
            list_of_strings.append(string[splitter:])
    if len(list_of_strings) is 0:
        list_of_strings.append(string)
    containing_empty = map(lambda x:x.strip(),list_of_strings)
    return "\n".join(list(filter(None,containing_empty)))

def replace_dict(known_codepoints):
    #replace unseen codepoints with a known one of the same utf8 category
    replace = {}
    for codepoint in known_codepoints:
        replace[unicodedata.category(chr(codepoint))] = codepoint
    return replace

def concatenate_input(input): # backward compatibility for older models
    return concatenate([input[0], input[1]], axis=-1)

def predict(input_path,model_folder_path,output_path,return_states,return_iob,return_lstmvis,split_iob):

    if not model_folder_path.endswith("/"):
        model_folder_path = model_folder_path + "/"

    if os.path.isfile(input_path):
        with open(input_path, "r", encoding="utf-8") as input_file:
            input_text=input_file.read()
    else:
        input_text = sys.stdin.read()
        if not input_text:
            print("No utf8 string from stdin and no input file found at: {}".format(input_path))
            sys.exit(1)

    model_meta = {}
    model_meta_path = model_folder_path+"model_meta.pickle"
    model_path = model_folder_path+"weights.hdf5"
    model_arch_path = model_folder_path+"state_model.h5"

    if os.path.isfile(model_path):
        model = load_model(model_path,custom_objects={"concatenate_input": concatenate_input,"concatenate": concatenate})
    else:
        print("No model found at: {}".format(model_path))
        sys.exit(2)
    
    if os.path.isfile(model_meta_path):
        with open(model_meta_path, "rb") as meta_file:
            model_meta.update(pickle.load(meta_file))
    else:
        print("No meta information found at: {}".format(model_meta_path))
        sys.exit(3)

    if split_iob not in model_meta["tag_to_indices"]:
        print("The model does not contain the given parameter for split_iob: {}".format(split_iob))
        sys.exit(4)

    if os.path.isfile(model_arch_path) and return_states:
        state_model = load_model(model_arch_path)
        state_model.load_weights(model_path, by_name=True)

    text_as_codepoints = []

    for character in input_text:
        text_as_codepoints.append(ord(character))

    sequence_length = model_meta["sequence_length"]
    sequences = []
    sequence_step = sequence_length
    sequence_step = 1 if return_states else sequence_step # as keras does not return the state at all timesteps

    for i in range(0, len(text_as_codepoints) - sequence_length + 1, sequence_step):
        sequences.append(i)
    replace = replace_dict(model_meta["codepoint_to_indices"].keys())
    x_unknown_cp = np.zeros((len(sequences), sequence_length, len(model_meta["codepoints_distinct"])), dtype=np.bool)
    x_unknown_cat = np.zeros((len(sequences), sequence_length, len(model_meta["categories_distinct"])), dtype=np.bool)
    
    for i, index in enumerate(sequences):
        for t, codepoint in enumerate(text_as_codepoints[index: index + sequence_length]):
            if codepoint in model_meta["codepoint_to_indices"]:
                x_unknown_cp[i, t, model_meta["codepoint_to_indices"][codepoint]] = 1
            else:
                if unicodedata.category(chr(codepoint)) in replace:
                    replacement = replace[unicodedata.category(chr(codepoint))]
                    x_unknown_cp[i, t, model_meta["codepoint_to_indices"][replacement]] = 1
                else:
                    x_unknown_cp[i, t, model_meta["codepoint_to_indices"][ord(" ")]] = 1
            if unicodedata.category(chr(codepoint)) in model_meta["category_to_indices"]:
                x_unknown_cat[i, t, model_meta["category_to_indices"][unicodedata.category(chr(codepoint))]] = 1
            else:
                x_unknown_cat[i, t, model_meta["category_to_indices"][unicodedata.category(" ")]] = 1
    #if the first two layers are of same instance type, we have two inputs
    if isinstance(model.layers[1],type(model.layers[0])):
        if return_states:
            _, states_h, states_c = state_model.predict([np.array(x_unknown_cp),np.array(x_unknown_cat)])
        else:
            tags_pred = model.predict([np.array(x_unknown_cp),np.array(x_unknown_cat)])
    else:
        #categories not included
        if return_states:
            _, states_h, states_c = state_model.predict(np.array(x_unknown_cp))
        else:
            tags_pred = model.predict(np.array(x_unknown_cp))

    if return_states:
        output_file = h5py.File(output_path, "w")
        output_file["hidden1"] = states_h
        output_file["cell1"] = states_c
        output_file.close()
        sys.exit(0)

    if len(tags_pred):
        tags_pred = list(itertools.chain.from_iterable(tags_pred.argmax(axis=-1)))
        tags_pred = list([model_meta["indices_to_tag"][pred] for pred in tags_pred])
        if return_iob:
            result = "category,codepoint,tag\n"
            for i, tag in enumerate(tags_pred):
                result += "{},{},{}\n".format(unicodedata.category(chr(text_as_codepoints[i])),text_as_codepoints[i],tag)
        elif return_lstmvis:
            result = " ".join(tags_pred)
        else:
            splitters = [i for i, j in enumerate(tags_pred) if j == split_iob]
            result = split_at_indices(input_text,splitters)
    else:
        result = input_text

    if not output_path:
        result_pre = "\/\/\/\n" #nasty because of theano/tf output. thanks for nothing
        print("".join([result_pre, result]))
    else:
        with open(output_path, "wb") as output_file:
            output_file.write(result.encode("utf-8"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmenter")

    parser.add_argument("-s", action="store", dest="split_iob", type=str, help="On which IOB-Tag a newline should split the output", default ="S",required=True)
    parser.add_argument("-m", action="store", dest="model_folder_path", type=str, help="Path to directory containing model files", default ="",required=True)
    parser.add_argument("-i", action="store", dest="input_path", type=str, help="Input file path if input is not given via stdin", default ="",required=False)
    parser.add_argument("-o", action="store", dest="output_path", type=str, help="Optional: output file path", default ="",required=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--states", dest="return_states",  default=False, action="store_true", help="Optional: output is lstm state vectors to h5 file",required=False)
    group.add_argument("--iob", dest="return_iob",  default=False, action="store_true", help="Optional: segmented output is formated in IOB style",required=False)
    group.add_argument("--lstmvis", dest="return_lstmvis",  default=False, action="store_true", help="Optional: ouput predicted tags space sperated for LSTMvis",required=False)

    args = parser.parse_args()

    predict(args.input_path,args.model_folder_path,args.output_path,args.return_states,args.return_iob,args.return_lstmvis,args.split_iob)