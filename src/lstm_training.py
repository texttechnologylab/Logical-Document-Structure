'''
Author: Rainer Killinger rainer@killinger.co
Model Training
'''

from __future__ import print_function
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten, Concatenate, concatenate, Input, Bidirectional, TimeDistributed, Merge, Lambda
from keras.utils.data_utils import get_file
from keras.utils import plot_model

from keras.models import Sequential, load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback, TensorBoard

from tensorflow import set_random_seed
from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import random
import sys
import pandas as pd
import itertools
import math

import plothelper
from generator import DocumentSequence


import os
import glob
import re
import pickle
from copy import deepcopy
import unicodedata
from scipy import stats

import argparse


np.random.seed(2342) # for reproducibility
set_random_seed(2342) # for reproducibility
# this is more or less useless when running via cuDNN, as its backward pass is by default non-deterministic.
# See http://deeplearning.net/software/theano/library/sandbox/cuda/dnn.html

parser = argparse.ArgumentParser(description='Character based LSTM segemend boundary detection model creator')
parser.add_argument('--cat', dest='use_category', default=False, action='store_true', help='Utilize category information in model')
parser.add_argument('-u', dest='lstm_units', type=int, default=128, help='Number of Units in LSTM')
parser.add_argument('-l', dest='sequence_length', type=int, default=60, help='Number of characters in one step')
parser.add_argument('-d', dest='lstm_dropout', type=float, default=0.0, help='LSTM dropout')
parser.add_argument('-b', dest='batch_size', type=int, default=128, help='Batch size')
parser.add_argument('-p', dest='name_prefix', type=str, default='model', help='Model name prefix')
parser.add_argument('-i', dest='iob_path', type=str, default='clauses.iob', help='Path to IOB training data')
parser.add_argument('-s', dest='sequence_step', type=int, default=1, help='Step size while training')


args = parser.parse_args()

# basic configuration
use_category = args.use_category
lstm_units = args.lstm_units
lstm_dropout = args.lstm_dropout
batch_size = args.batch_size
sequence_length = args.sequence_length
sequence_step = args.sequence_step 

epochs = 4
validation_split = 0.1
test_split = 0.1

usescat = "_cat" if use_category else ""
modelname = "modelzoo/{}_sl{}_units{}_dropout{}{}".format(args.name_prefix,str(sequence_length),lstm_units,str(int(lstm_dropout*10)).zfill(2),usescat)
print(modelname)

#prepare folder structure
if not os.path.exists(modelname + '/checkpoints'):
    os.makedirs(modelname + '/checkpoints')

if not os.path.isfile(args.iob_path):
    print("IOB file not found at given path")
    sys.exit(1)

with open(args.iob_path) as csv_file:
    df = pd.read_csv(csv_file)
    codepoints = df['codepoint'].values
    categories = df['category'].values
    tags = df['tag'].values

test_split_amount = int(math.ceil(len(codepoints)*test_split))

test_codepoints = codepoints[-test_split_amount:]
test_categories = categories[-test_split_amount:]
test_tags = tags[-test_split_amount:]

validation_split = len(codepoints)*validation_split
validation_split = (validation_split/(len(codepoints)-test_split_amount))

# check if we can pickup from checkpoint
# assume that we can continue with latest checkpoint this will later on fail if dimensions are different
epochs_ran = 0
if len(list(glob.iglob(modelname + '/checkpoints/*.hdf5'))):
    latest_checkpoint = sorted(glob.glob(modelname + '/checkpoints/*.hdf5'), key=os.path.getmtime, reverse=True)
    if latest_checkpoint is not None:
        print("took : "+str(latest_checkpoint[0]))
        match = re.search('(?<=-)\d+',latest_checkpoint[0])
        if match.group(0) is not None:
            epochs_ran = int(match.group(0)) + 1 # offset as we started with 0
            if epochs_ran == epochs:
                print('Nothing to do as {} of {} epochs already finished'.format(str(epochs_ran),str(epochs)))
                sys.exit(0)
            print('Picking up at model weigths at epoch',epochs_ran)
            print('This is prone to errors or fails if model has changed!')

print('Corpus length:', len(codepoints) ,' characters')

codepoints_distinct = sorted(list(set(codepoints)))
codepoint_to_indices = dict((c, i) for i, c in enumerate(codepoints_distinct))
indices_to_codepoint = dict((i, c) for i, c in enumerate(codepoints_distinct))
print('Number of characters:', len(codepoints_distinct))

categories_distinct = sorted(list(set(categories)))
category_to_indices = dict((c, i) for i, c in enumerate(categories_distinct))
indices_to_category = dict((i, c) for i, c in enumerate(categories_distinct))
print('Number of categories:', len(categories_distinct))

tags_distinct = sorted(list(set(tags)))
tag_to_indices = dict((c, i) for i, c in enumerate(tags_distinct))
indices_to_tag = dict((i, c) for i, c in enumerate(tags_distinct))
print('Number of tags:', len(tags_distinct))

# remove test from training
categories = categories[:len(codepoints)-test_split_amount]
tags = tags[:len(codepoints)-test_split_amount]
codepoints = codepoints[:len(codepoints)-test_split_amount]

# chop input into redundant sequences of sequence_length characters
print('Generating sequences of length:',sequence_length)

#training set
sequences = []
for i in range(0, len(codepoints) - sequence_length+1, sequence_step):
    sequences.append(i) #just generating window indexes
print('Number of sequences:', len(sequences))

print('Creating tensors from data...')

#test set
test_sequences = []
test_pred_tags = []
for i in range(0, len(test_codepoints) - sequence_length+1, sequence_length):
    test_sequences.append(i)
    test_pred_tags.append(test_tags[i: i + sequence_length])

x_test_cp = np.zeros((len(test_sequences), sequence_length, len(codepoints_distinct)), dtype=np.bool)
x_test_cat = np.zeros((len(test_sequences), sequence_length, len(categories_distinct)), dtype=np.bool)
y_test = np.zeros((len(test_sequences), sequence_length, len(tags_distinct)), dtype=np.bool)

for i,index in enumerate(test_sequences):
    for j, codepoint in enumerate(test_codepoints[index: index + sequence_length]):
        x_test_cp[i, j, codepoint_to_indices[codepoint]] = 1
        if use_category:
            x_test_cat[i, j, category_to_indices[unicodedata.category(chr(codepoint))]] = 1
    for  j, tag in enumerate(test_tags[index: index + sequence_length]):
        y_test[i, j, tag_to_indices[tag]] = 1


# setup model: 2 layer BiLSTM with the ability to get hidden and cell state.
print('Creating model...')

def concatenate_input(input):
    return concatenate([input[0], input[1]], axis=-1)

input_layer_cp = Input(batch_shape=(None,None,len(codepoints_distinct)), name='input_cp')

if use_category:
    input_layer_cat = Input(batch_shape=(None,None,len(categories_distinct)), name='input_cat')
    input_layer = [input_layer_cp, input_layer_cat]
    concatenated_input = concatenate([input_layer_cp, input_layer_cat], axis=-1)
    bilstm = Bidirectional(LSTM(units=lstm_units,dropout=lstm_dropout, return_sequences=True, implementation=2, name='bilstm'))(concatenated_input)
   
else:
    input_layer = input_layer_cp
    bilstm = Bidirectional(LSTM(units=lstm_units,dropout=lstm_dropout, return_sequences=True, implementation=2, name='bilstm'))(input_layer_cp)

lstm_fwd_out, fwd_state_h, fwd_state_c = LSTM(units=lstm_units, return_state=True, return_sequences=True, implementation=2, name='lstm_fwd')(bilstm)
lstm_bwd_out, bwd_state_h, bwd_state_c = LSTM(units=lstm_units, return_state=True, return_sequences=True, implementation=2, name='lstm_bwd')(bilstm)  

print(fwd_state_h.shape)
print(fwd_state_c.shape)

merged_output = concatenate([lstm_fwd_out, lstm_bwd_out], name='bilstm_output')
merged_state_h = concatenate([fwd_state_h, bwd_state_h], name='state_h')
merged_state_c = concatenate([fwd_state_c, bwd_state_c], name='state_c')

time_dist_activation = TimeDistributed(Dense(len(tags_distinct), activation='softmax'))(merged_output)

model_vis = Model(inputs=input_layer, outputs=[merged_output, merged_state_h, merged_state_c])
model = Model(inputs=input_layer, outputs=[time_dist_activation])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])


# acutally pickup model weigths
if len(list(glob.iglob(modelname + '/checkpoints/*.hdf5'))):
    latest_checkpoint = sorted(glob.glob(modelname + '/checkpoints/*.hdf5'), key=os.path.getmtime, reverse=True)
    if latest_checkpoint is not None:
        model.load_weights(latest_checkpoint[0])

print('Batch size:',batch_size)
print('Validation split:',validation_split)
print('There are',epochs-epochs_ran,'epochs left')
print(model.summary())
print(model_vis.summary())
plot_model(model, to_file=modelname + '/model_visualization_macro.pdf',show_shapes=True, show_layer_names=True)

#save state returning model 
model_vis.save(modelname + '/state_model.h5')

#pickup or create evaluation history
accumulated_history =  {'acc' : [0.0], 'val_acc':[0.0], 'loss' : [0.0], 'val_loss':[0.0],'cm':[],'cr':[],'tags':tags_distinct}
if os.path.isfile(modelname + '/checkpoints/evaluation.pickle'):
    with open(modelname + '/checkpoints/evaluation.pickle', 'rb') as eval_file:
        accumulated_history.update(pickle.load(eval_file))

model_meta = {'categories_distinct':categories_distinct,'category_to_indices':category_to_indices,'indices_to_category':indices_to_category,'codepoints_distinct':codepoints_distinct,'codepoint_to_indices':codepoint_to_indices,'indices_to_codepoint':indices_to_codepoint,'tags_distinct':tags_distinct,'tag_to_indices':tag_to_indices,'indices_to_tag':indices_to_tag,'sequence_length':sequence_length}


for epoch in range(epochs_ran,epochs):
    #callbacks
    checkpoint_path='{}{}{}'.format(modelname + '/checkpoints/weights-', epoch, '-{val_categorical_accuracy:.5f}.hdf5')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', period=1)
    best_model = ModelCheckpoint(filepath=modelname + '/weights.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto')
    callbacks = [checkpoint,best_model,early_stop]
    
    #generator_fit 
    validation_split_index = int(len(sequences) * (1.0-validation_split))
    training_generator =  DocumentSequence(batch_size,sequences[:validation_split_index],codepoints,categories,tags,model_meta,use_category)
    validation_generator =  DocumentSequence(batch_size,sequences[validation_split_index:],codepoints,categories,tags,model_meta,use_category)
    history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                                use_multiprocessing=False, workers=1, max_queue_size=20, epochs=1, shuffle=False, callbacks=callbacks, verbose=1)

    #save meta here to make sure fit succeeded
    with open(modelname + '/model_meta.pickle', 'wb') as meta_file:
        pickle.dump(model_meta,meta_file)

    #print(history.history)
    accumulated_history['acc'].extend(history.history['categorical_accuracy'])
    accumulated_history['val_acc'].extend(history.history['val_categorical_accuracy'])
    accumulated_history['loss'].extend(history.history['loss'])
    accumulated_history['val_loss'].extend(history.history['val_loss'])

    if use_category:
        tags_pred = model.predict([np.array(x_test_cp),np.array(x_test_cat)])
    else:
        tags_pred = model.predict([np.array(x_test_cp)])

    tags_pred = list(itertools.chain.from_iterable(tags_pred.argmax(axis=-1)))
    tags_pred = list([indices_to_tag[pred] for pred in tags_pred])

    tags_true = test_pred_tags
    tags_true = list(itertools.chain.from_iterable(tags_true))

    accumulated_history['cm'].extend([confusion_matrix(tags_true, tags_pred)])
    accumulated_history['cr'].extend([classification_report(tags_true, tags_pred,target_names=accumulated_history['tags'])])

    #save history
    with open(modelname + '/checkpoints/evaluation.pickle', 'wb') as eval_file:
        pickle.dump(accumulated_history,eval_file)

    # ad epoch to plot
    plothelper.pdf_plot(accumulated_history,modelname)
