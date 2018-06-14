# coding=utf-8
'''
Author: Rainer Killinger rainer@killinger.co
This script creates swarm plots for a csv (name,value)
grouping it by a substring (of its filename) and compares this group internaly
given another substring (of its filename) 
'''

import matplotlib
matplotlib.use('Agg') #no display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr,binom_test
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import seaborn as sns
import csv
import sys
import re
import argparse
from pathlib import Path


def resolveNamingScheme(name):
    properties = {"sequence_length":0, "units":0, "dropout":0, "uses_categories": False}
    if "_cat" in name:
        properties["uses_categories"] = True
        match = re.search('(?<=dropout).*(?=\_cat)', name)
    else:
        match = re.search('(?<=dropout).*', name)
    properties["dropout"] = float(match.group(0))*0.1 # shift decimal point to actual position

    match = re.search('(?<=sl).*(?=\_units)', name)
    properties["sequence_length"] = int(match.group(0))

    match = re.search('(?<=units).*(?=\_dropout)', name)
    properties["units"] = int(match.group(0))

    return properties


def plotSwarm(path,dataframe):
    dataframe.columns = ["Sequenzlänge", "Zelleinheiten", "Dropout", "Nutzt Unicode-\nKategorien", "F₁-Wert"] # germanize

    result_path =  "{}/{}_units.pdf".format(path.parent,path.stem)
    with PdfPages(result_path) as pdf:
        dataframe.rename(columns={"F₁-Wert": "FWert"}, inplace=True) #statsmodels is ascii only
        spear = spearmanr(dataframe["Zelleinheiten"],dataframe["FWert"], axis=None)
        print("--- Units ---")
        print(spear)
        mod = ols("FWert ~ Zelleinheiten",data=dataframe).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
        print("η² {}".format(str(esq_sm)))
        spear_p = "{0:.4e}".format(float(spear[1])) if float(spear[1])<0.0001 else "{0:.4f}".format(float(spear[1]))
        print("tex: Zelleinheiten & ${0:.4f}$ & ${1}$ & ${2:.4f}$ \\\\".format(float(spear[0]),spear_p,float(esq_sm)))
        dataframe.rename(columns={"FWert": "F₁-Wert"}, inplace=True)
        ax = sns.factorplot(x="Zelleinheiten", y="F₁-Wert", hue="Nutzt Unicode-\nKategorien", data=dataframe, col="Sequenzlänge",
                            palette="hls", dodge=True, kind="swarm", aspect=0.5)
        pdf.savefig()
        plt.close()


    result_path =  "{}/{}_dropout.pdf".format(path.parent,path.stem)
    with PdfPages(result_path) as pdf:
        dataframe.rename(columns={"F₁-Wert": "FWert"}, inplace=True) #statsmodels is ascii only
        spear = spearmanr(dataframe["Dropout"],dataframe["FWert"], axis=None)
        print("--- Dropout ---")
        print(spear)
        mod = ols("FWert ~ Dropout",data=dataframe).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
        spear_p = "{0:.4e}".format(float(spear[1])) if float(spear[1])<0.0001 else "{0:.4f}".format(float(spear[1]))
        print("η² {}".format(str(esq_sm)))
        print("tex: Dropout & ${0:.4f}$ & ${1}$ & ${2:.4f}$ \\\\".format(float(spear[0]),spear_p,float(esq_sm)))
        dataframe.rename(columns={"FWert": "F₁-Wert"}, inplace=True)
        ax = sns.swarmplot(x="Dropout", y="F₁-Wert", hue="Nutzt Unicode-\nKategorien", data=dataframe,
                            palette="hls", dodge=True)
        pdf.savefig()
        plt.close()


    result_path =  "{}/{}_sequence.pdf".format(path.parent,path.stem)
    with PdfPages(result_path) as pdf:
        dataframe.rename(columns={"F₁-Wert": "FWert"}, inplace=True) #statsmodels is ascii only
        spear = spearmanr(dataframe["Sequenzlänge"],dataframe["FWert"], axis=None)
        print("--- Sequence length ---")
        print(spear)
        mod = ols("FWert ~ Sequenzlänge",data=dataframe).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
        print("η² {}".format(str(esq_sm)))
        spear_p = "{0:.4e}".format(float(spear[1])) if float(spear[1])<0.0001 else "{0:.4f}".format(float(spear[1]))
        print("tex: Sequenzlänge & ${0:.4f}$ & ${1}$ & ${2:.4f}$ \\\\".format(float(spear[0]),spear_p,float(esq_sm)))
        dataframe.rename(columns={"FWert": "F₁-Wert"}, inplace=True)
        ax = sns.swarmplot(x="Sequenzlänge", y="F₁-Wert", hue="Nutzt Unicode-\nKategorien", data=dataframe,
                            palette="pastel", dodge=True)
        pdf.savefig()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Character based LSTM segemend boundary detection model creator')
    parser.add_argument('-i', dest='result_csv_path', type=str, help='Path to csv file cointaining results in model-name,score format',required=True)
    args = parser.parse_args()
    
    results = {}
    df = pd.DataFrame(columns=["sequence_length", "units", "dropout", "uses_categories", "f1_score"])
    
    with open(args.result_csv_path) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip header
        for row in reader:
            results[row[0]] = row[1]
    
    if sys.version_info[0] <= 3: # iterating key,value is different in py2 py3
        for name,score in results.items():
            properties = resolveNamingScheme(name)
            properties["f1_score"] = float(score)
            df = df.append(properties, ignore_index=True)
    else:
        for name,score in results.iteritems():
            properties = resolveNamingScheme(name)
            properties["f1_score"] = float(score)
            df = df.append(properties, ignore_index=True)

    df_nocat = df[(df.uses_categories == False)]
    cat_improved = 0
    for index in range(0,df_nocat.shape[0]-1):
        row = (df_nocat.iloc[[index]])
        counter_part = df[(df.uses_categories == True) & (df.sequence_length == row.sequence_length.values[0])]
        counter_part = counter_part[(df.units == row.units.values[0]) & (df.dropout == row.dropout.values[0])].iloc[[0]]
        if counter_part.f1_score.values[0] > row.f1_score.values[0]:
            cat_improved+=1

    print(cat_improved)
    print(df_nocat.shape[0])
    print(binom_test(cat_improved, n=df_nocat.shape[0], p=0.5, alternative='greater'))

    path = Path(args.result_csv_path)
    plotSwarm(path,df)


if __name__ == '__main__':
    main()
