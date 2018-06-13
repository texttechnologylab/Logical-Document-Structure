'''
Author: Rainer Killinger rainer@killinger.co
This script adds utf8 category information if it is missing
'''

import csv
import sys
import unicodedata

with open("YOUR_IOB_CSV_HERE.iob") as csvfile:
    newCSV = ""
    reader = csv.reader(csvfile)
    skippedHeader = False
    for row in reader:
        if not skippedHeader:
            newCSV += ",".join(row) + "\n"
            skippedHeader = True
            continue
        newCSV += "{},{},{}\n".format(unicodedata.category(chr(int(row[1]))),row[1],row[2])

    with open("enriched_with_cat.iob", "w") as f:
        f.write(newCSV)
