#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment
# Pipeline Part 5: Scoring
# This script scores the classifiers

import argparse
import json
import os
import re
from sklearn.metrics import precision_recall_fscore_support as scorer
from sklearn.metrics import classification_report
import sys

parser = argparse.ArgumentParser()
parser.add_argument("test_data", help = "formatted test data filename")
parser.add_argument("predictions", help = "predicted classification data filename")
args = parser.parse_args()

with open(args.test_data, "r") as f:
    true_data = json.load(f)

with open(args.predictions, "r") as f:
    predicted_data = json.load(f)

true_labels = []

for obs in true_data:
    true_labels.append(obs[1]["NEtag"])

predicted_me = []
predicted_nb = []
predicted_dt = []

for obs in predicted_data:
    predicted_me.append(obs[1]["me_pred"])
    predicted_nb.append(obs[1]["nb_pred"])
    predicted_dt.append(obs[1]["dt_pred"])

for labels in [predicted_me, predicted_nb, predicted_dt]:
    print(scorer(true_labels, labels, average = "micro"))

    print(classification_report(true_labels, labels))