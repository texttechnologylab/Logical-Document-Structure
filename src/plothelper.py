'''
Author: Rainer Killinger rainer@killinger.co
plots model history
'''

from sklearn.metrics import confusion_matrix,classification_report
import numpy as np
import random
import sys
import pandas as pd
import itertools

import pydot
import matplotlib
matplotlib.use('Agg') #no display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

def pdf_plot(history,modelname):
    with PdfPages(modelname + '/evaluation-intrinsic.pdf') as pdf:
        # Accuracy
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(min(min(history['acc'][1:]),min(history['acc'][1:])),)
        ax.set_xlim(1,len(history['acc'])-1)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model categorical accuracy') 
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        pdf.savefig()
        plt.close()
        # Loss
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(1,len(history['loss'])-1)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        pdf.savefig()
        plt.close()
        # CMs & CRs
        for i,cmatrix in enumerate(reversed(history['cm'])):
            plt.figure()
            j = len(history['cm']) - i #human numbering
            plt.subplot(1, 2, 1)
            plot_confusion_matrix(cm=cmatrix,classes=history['tags'],normalize=False,title='Epoch %s confusion matrix' % j)
            plt.subplot(1, 2, 2)
            plot_confusion_matrix(cm=cmatrix,classes=history['tags'],normalize=True,title='')
            pdf.savefig()
            plt.close()

            plt.figure()
            plot_classification_report(cr=history['cr'][::-1][i], with_avg_total=True)

            plt.subplots_adjust(left=0.2, bottom=0.2)
            pdf.savefig()
            plt.close()

#adapted from sklearn examples
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest',cmap=cmap)

    if normalize:
        title = '{} normalized'.format(title)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if normalize is False:
        plt.ylabel('True tag')
    plt.xlabel('Predicted tag')

def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')
    classes = []
    table_text = []
    for line in lines[2 : (len(lines) - 3)]:
        t = line.split()
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        table_text.append(v)

    if with_avg_total:
        avg_total = lines[len(lines) - 1].split()
        classes.append('Avg / Total')
        avg_total = [float(x) for x in t[1:len(avg_total) - 1]]
        table_text.append(avg_total)

    fig, ax = plt.subplots()

    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=table_text,rowLabels=classes,colLabels=['Precision', 'Recall', 'F1-score'], loc='center')

    fig.tight_layout()
