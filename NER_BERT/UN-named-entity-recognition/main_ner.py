#!/usr/bin/env python3

# Leslie Huang
# Pipeline Part 4: NER
# This script runs an ensemble of classifiers for NER

from sentences import Sentence, saturatedToken, Token, POSTaggedSentence, ChunkTaggedSentence, saturatedSentence
from taggers import ClassifierData

import argparse
import itertools
import json
from multiprocessing import Pool
import nltk
from nltk.classify import MaxentClassifier
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("training_dir", help = "dir to read preppedforNERSentences JSON files from")
parser.add_argument("test_dir", help = "dir to read preppedforNERSentences training JSON files from")
parser.add_argument("n_iterations", help = "num iters for MaxEnt Classifier")
parser.add_argument("output_dir", help = "dir to write output to")
parser.add_argument("output_file", help = "file to write all results to")
parser.add_argument("test_output", help = "file to write test data to (for scoring)")
args = parser.parse_args()


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


def METagger(training_data, test_data, n_iter, output_dir):
    """
    Runs METagger in a mp.pool
    """
    pool = Pool(os.cpu_count() - 1)

    me_classifier = MaxentClassifier.train(training_data, max_iter = n_iter)
    predictions = []

    with open(os.path.join(output_dir, "me_pred.json"), 'w') as f:

        for index, result in enumerate(pool.imap(me_classifier.classify, test_data)):
            prediction = label_test_data(test_data[index], result)
            predictions.append(prediction)

            json.dump(prediction, f)
            f.write("\n")

    return predictions


def NBTagger(training_data, test_data, output_dir):
    pool = Pool(os.cpu_count() - 1)

    nb_classifier = nltk.NaiveBayesClassifier.train(training_data)
    predictions = []

    with open(os.path.join(output_dir, "nb_pred.json"), 'w') as f:

        for index, result in enumerate(pool.imap(nb_classifier.classify, test_data)):
            prediction = label_test_data(test_data[index], result)
            predictions.append(prediction)

            json.dump(prediction, f)

    return predictions


def DecisionTagger(training_data, test_data, output_dir):

    pool = Pool(os.cpu_count() - 1)

    decision_classifier = nltk.DecisionTreeClassifier.train(training_data)

    predictions = []

    with open(os.path.join(output_dir, "dt_pred.json"), 'w') as f:

        for index, result in enumerate(pool.imap(decision_classifier.classify, test_data)):
            prediction = label_test_data(test_data[index], result)
            predictions.append(prediction)

            json.dump(prediction, f)
            f.write("\n")

    return predictions


def label_test_data(word_dict, predicted_classes):
    """
    Helper function to format Classifier results for output
    """

    return (word_dict, predicted_classes)


if __name__ == '__main__':

    # Load the test and training data, and output test data for scoring
    classifier_data = ClassifierData.load_and_format_data(args.training_dir,
        args.test_dir, args.test_output)

    # run the classifiers
    me_pred = METagger(classifier_data.training_data, classifier_data.test_data, int(args.n_iterations), args.output_dir)

    dt_pred = DecisionTagger(classifier_data.training_data, classifier_data.test_data, args.output_dir)

    nb_pred = NBTagger(classifier_data.training_data, classifier_data.test_data, args.output_dir)

    # output data
    out_data = []

    for index, word_dict in enumerate(classifier_data.test_data):
        out = [word_dict,
            {
                "me_pred": me_pred[index][1],
                "dt_pred": dt_pred[index][1],
                "nb_pred": nb_pred[index][1],
            }
        ]

        out_data.append(out)

    with open(args.output_file, "w") as f:
        json.dump(out_data, f, indent = 2)


