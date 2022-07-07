#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import copy
import itertools
import json
from multiprocessing import Pool
import nltk
from nltk.chunk import tree2conlltags, conlltags2tree
from nltk.corpus import brown
from nltk.corpus import conll2000
from nltk.classify import MaxentClassifier
import os
import re
import sys
import time

nltk.download("brown") # for POS tagging
nltk.download("conll2000") # for chunking

class POSTagger:

    def __init__(self, re_expressions):
        self.re_expressions = re_expressions # regex patterns for the RegexpTagger

    def tagger(self, sentence_obj):
        """
        This method follows the NLTK Book, Chapter 5: http://www.nltk.org/book/ch05.html
        Bigram, unigram, and regex taggers are used before the default tagger assigns
        the most frequent tag from the Brown corpus
        Args:
            sentence_obj : Sentence object
        Returns:
            list of (word, tag) tuples
        """
        trained_sentences = brown.tagged_sents()

        defaultTagger = nltk.DefaultTagger(self.get_default_tag(trained_sentences))
        reTagger = nltk.RegexpTagger(self.re_expressions, backoff = defaultTagger)
        uniTagger = nltk.UnigramTagger(trained_sentences, backoff = reTagger)
        biTagger = nltk.BigramTagger(trained_sentences, backoff = uniTagger)

        return biTagger.tag(sentence_obj.list_of_words)

    def get_default_tag(self, trained_sentences):
        """
        Helper function for tagger. Finds the most frequent tag from the corpus, to use as default
        Args:
            trained_sentences : trained nltk sentences
        Returns:
            str of most frequent tag
        """
        freq_dist = nltk.FreqDist(sent[0][1] for sent in trained_sentences)

        return max(freq_dist)


class UnigramChunker(nltk.ChunkParserI):
    """
    Label sentence with chunk tags based on POS tags
    Methods taken from the NLTK book, Chapter 7: https://www.nltk.org/book/ch07.html
    """

    def __init__(self, train_sents):
        self.train_sents = train_sents

        self.unitagger = nltk.UnigramTagger(self.train_sents)

    @classmethod
    def read_from_json(cls, json_chunked_data_fn):
        """
        Load train_sents from a JSON file of manually tagged chunk data
        """

        with open(json_chunked_data_fn, "r") as f:
            data = json.load(f)

        # convert inner list to tuples of (W,POS,CHUNK) and keep only POS,CHUNK
        all_tags = [[tuple(w) for w in s] for s in data]
        train_sents = [[(p,c) for w,p,c in s] for s in all_tags]

        return cls(train_sents)

    @classmethod
    def load_nltk_chunked_sentences(cls):
        """
        Load CONLL2000 chunktagged sentences and convert from nltk.tree format
        Returns:
            List of lists where inner list is [(pos, chunk_tag), ... ] representing one sentence
        """
        train_sents = [
            [(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in nltk.chunk.tree2conlltags(sentence)]
            for sentence in conll2000.chunked_sents()
                ]

        return cls(train_sents)


    def tag_sentences(self, posTaggedSentence):
        """
        Chunktag a posTaggedSentence object
        """
        pos_tags = [pos for (word, pos) in posTaggedSentence.pos]

        pos_and_chunk_tags = self.unitagger.tag(pos_tags)

        # # Use this block to output word,pos,chunk tags
        # chunk_tags = [chunk for (pos, chunk) in pos_and_chunk_tags]
        # words = [word for (word, pos) in posTaggedSentence.pos]
        # all_tags = list(zip(words, pos_tags, chunk_tags))
        # return all_tags

        return pos_and_chunk_tags

# class NEChunker():
#     """
#     Uses the built-in NLTK NE tagger (will generate a baseline of NEtags)
#     """
#     def __init__(self, posTaggedSentence):
#         tree = nltk.ne_chunk(posTaggedSentence.pos)
#         iob_tags = tree2conlltags(tree)

#         # insert code to change IOB tag format

#         self.ne_tags = tree2conlltags(tree)




# Helper function for ClassifierData class

def load_from_json(dirname):
    """
    Load data from JSON containing dicts
    """
    data = []
    for fn in os.listdir(dirname):
        with open(os.path.join(dirname, fn), "r") as f:
            sentence = json.load(f)

            data += sentence

    return data


class ClassifierData(object):

    def __init__(self, training_data, test_data, test_output_target):
        self.training_data = training_data
        self.test_data = test_data
        self.test_output_target = test_output_target

    @classmethod
    def load_and_format_data(cls, training_dirname, test_dirname, test_output_target):
        """
        Load training and test data from JSON and format them for MaxEntClassifier
        """
        training = load_from_json(training_dirname)
        test = load_from_json(test_dirname)

        # Make a copy of test data and write version formatted for scoring
        test_for_output = copy.deepcopy(test)
        output = []

        for w in test_for_output:
            NEtag = w.pop("NEtag")
            output.append(
                [w, {"NEtag": NEtag}]
                )

        with open(test_output_target, "w") as f:
            json.dump(output, f, indent = 2)

        me_training = []

        for word_dict in training:
            NEtag = word_dict.pop("NEtag")
            me_training.append(
                (word_dict, NEtag)
                )

        me_test = []

        for word_dict in test:
            NEtag = word_dict.pop("NEtag")
            me_test.append(word_dict)

        return cls(me_training, me_test, test_output_target)
