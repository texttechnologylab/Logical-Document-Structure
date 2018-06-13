'''
Modified decorator version of example https://github.com/keras-team/keras/issues/1638#issuecomment-182139908
'''
import numpy as np
import random
import sys
import pandas as pd
import itertools
import math
import threading
import unicodedata
from keras.utils import Sequence

class DocumentSequence(Sequence):

    def __init__(self,batch_size,sequences,codepoints,categories,target_tags,model_meta,use_category):
        self.batch_size = batch_size
        self.sequences = sequences
        self.codepoints = codepoints
        self.categories = categories
        self.target_tags = target_tags
        self.model_meta = model_meta
        self.use_category = use_category
        self.x_cp = np.zeros((batch_size, model_meta["sequence_length"], len(model_meta["codepoint_to_indices"])), dtype=np.bool)
        self.x_cat = np.zeros((batch_size,model_meta["sequence_length"], len(model_meta["category_to_indices"])), dtype=np.bool)
        self.y = np.zeros((batch_size, model_meta["sequence_length"], len(model_meta["tag_to_indices"])), dtype=np.bool)


    def __len__(self):
        return int(np.ceil(len(self.sequences) / self.batch_size))

    def __getitem__(self, idx):
        document_string_index = int(idx * self.batch_size)
        self.x_cp = np.zeros((self.batch_size, self.model_meta["sequence_length"], len(self.model_meta["codepoint_to_indices"])), dtype=np.bool)
        self.x_cat = np.zeros((self.batch_size,self.model_meta["sequence_length"], len(self.model_meta["category_to_indices"])), dtype=np.bool)
        self.y = np.zeros((self.batch_size, self.model_meta["sequence_length"], len(self.model_meta["tag_to_indices"])), dtype=np.bool)

        for i, sequence_i in enumerate(self.sequences[document_string_index:]):
            batch_i = i%self.batch_size
            if batch_i == 0 and i>0:
                if self.use_category:
                    return [self.x_cp,self.x_cat], self.y
                else:
                    return [self.x_cp], self.y
            
            context_i = 0
            sent=""
            for j in range(sequence_i,sequence_i+self.model_meta["sequence_length"]): 
                sent+=chr(self.codepoints[j])
                self.x_cp[batch_i,context_i, self.model_meta["codepoint_to_indices"][self.codepoints[j]]] = 1
                self.x_cat[batch_i,context_i, self.model_meta["category_to_indices"][self.categories[j]]] = 1
                self.y[batch_i,context_i, self.model_meta["tag_to_indices"][self.target_tags[j]]] = 1
                context_i += 1
