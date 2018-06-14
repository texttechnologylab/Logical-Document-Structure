# coding=utf-8

"""
Author: Rainer Killinger rainer@killinger.co
given an iob file it generatats necessary files for LSTMvis.
the last thing missing is to get the states file via segmenter.py
"""

import csv
import sys
import os 
import unicodedata
import argparse
import pickle
import subprocess

sequence_of_codepoints = []
dict_of_codepoints = {}

sequence_of_categories = []
dict_of_categories = {}
dict_categories_used = 0

sequence_of_tags = []
dict_of_tags = {}
dict_tags_used = 0


parser = argparse.ArgumentParser(description="")
parser.add_argument("-i", dest="iob_path", type=str, default="clauses.iob", help="IOB file input path", required=True)
parser.add_argument("-o", dest="output_path", type=str, default="result", help="Optional: folder path for output files", required=False)
args = parser.parse_args()

if not args.output_path.endswith("/"):
    args.output_path = args.output_path  + "/"
if not os.path.exists(args.output_path) :
    os.makedirs(args.output_path)

with open(args.iob_path, "r") as f:
    reader = csv.reader(f)
    skippedHeader = False
    for row in reader:
        if len(row[0])>4:
            continue

        category_string = row[0]
        codepoint = int(row[1])
        iob_tag = row[2]

        if not category_string in dict_of_categories:
            dict_of_categories[category_string] = dict_categories_used
            dict_categories_used+=1

        if not iob_tag in dict_of_tags:
            dict_of_tags[iob_tag] = dict_tags_used
            dict_tags_used+=1

        sequence_of_categories.append(category_string)
        sequence_of_tags.append(iob_tag)
        sequence_of_codepoints.append(chr(codepoint))
        dict_of_codepoints[codepoint] = chr(codepoint)


dict_of_categories = dict((v,k) for k, v in dict_of_categories.items())
dict_of_tags = dict((v,k) for k, v in dict_of_tags.items())

# replace space " " (\u32) with utf8  so called punctuation space " " (\u8200) !different!
# we need this so lstmvis can display some kind spaces 
dict_of_codepoints[32] = chr(8200)
dict_of_codepoints[10] = chr(8200)
#print(dict_of_codepoints)
#path to folder containing scripts
script_path =os.path.dirname(os.path.abspath(__file__))
if not script_path.endswith("/"):
    script_path = script_path  + "/"

# txt_to_hdf5_dict.py does this
#with open(args.output_path+"train.dict", "wb") as f:
#    for key,value in dict_of_codepoints.items():
#        f.write("{} {}\n".format(value,key).encode('utf-8'))
#    f.close()

#with open(args.output_path+"iob.dict", "wb") as f:
#    for key,value in dict_of_tags.items():
#        f.write("{} {}\n".format(value,key).encode('utf-8'))
#    f.close()

#with open(args.output_path+"cat.dict ", "wb") as f:
#    for key,value in dict_of_categories.items():
#        f.write("{} {}\n".format(value,key).encode('utf-8'))
#    f.close()

with open(args.output_path+"chars.seq.txt", "wb") as f:
    f.write("".join(list(str(item)for item in sequence_of_codepoints)).encode('utf-8'))
    f.close()

with open(args.output_path+"iob.seq.txt", "wb") as f:
    f.write(" ".join(list(str(item)for item in sequence_of_tags)).encode('utf-8'))
    f.close()

with open(args.output_path+"cat.seq.txt", "wb") as f:
    f.write(" ".join(list(str(item)for item in sequence_of_categories)).encode('utf-8'))
    f.close()

subprocess.run(["python3", script_path+"txt_to_hdf5_dict.py", args.output_path+"chars.seq.txt", args.output_path+"train" , "-c"], check=True)

for filename in ["iob","cat"]:
    subprocess.run(["python3", script_path+"txt_to_hdf5_dict.py", args.output_path+filename+".seq.txt", args.output_path+filename], check=True)
