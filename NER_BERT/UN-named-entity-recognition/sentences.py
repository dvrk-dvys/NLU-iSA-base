#!/usr/bin/env python3

# Leslie Huang
# Natural Language Processing Final Assignment

import argparse
import itertools
from geotext import GeoText
import json
from multiprocessing import Pool
import nltk
import os
import re
import sys
import time


# Helper functions for RawDocument class
def split_lines_into_token_tag(raw_text):
    """
    Helper function for group_text_into_sentences. Converts rawdata to List of Lists
    Args:
        raw_text : list of lines, each of which is word\tNEtag
    Returns:
        [ [word, NEtag], [word, NEtag], "\n", ... ], where newlines designate end of sentence
    """
    return [line if line == "\n" else line.rstrip().split("\t") for line in raw_text]

def group_text_into_sentences(raw_text):
    """
    Returns List of Lists where inner lists are sentences w/ or w/o NEtags. Thanks itertools!
    Args:
        raw_data : list of lines
    Returns:
        List of lists, where each inner list is a sentence of (word, NEtag) tuples
    """
    grouped_by_sentence = []

    for _, g in itertools.groupby(split_lines_into_token_tag(raw_text), lambda x: x == "\n"):
        grouped_by_sentence.append(list(g)) # Store group iterator as a list

    return [g for g in grouped_by_sentence if g != ["\n"]]


class RawDocument(object):
    """
    Initial container for a single (training or test) text.
    """
    def __init__(self, is_training, filepath):
        self.is_training = is_training
        self.raw_text = self.load(filepath) # Is a list of lines from the text
        self.text_grouped_into_sentences = group_text_into_sentences(self.raw_text)

    def load(self, filepath):
        with open(filepath, "r") as f:
            raw_text = f.readlines()

        return raw_text


class Token(object):
    """
    Container for a single token and its features
    """
    def __init__(self, token, sentence_position, sentence_start, sentence_end, NEtag):
        self.token = token
        self.sentence_position = sentence_position
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end
        self.NEtag = NEtag


class saturatedToken(object):
    def __init__(self, token_obj, countries_list, orgs_list, progs_list):
        self.token = token_obj.token
        self.sentence_position = token_obj.sentence_position
        self.sentence_start = token_obj.sentence_start
        self.sentence_end = token_obj.sentence_end
        self.NEtag = token_obj.NEtag

        self.case = "lower" if self.token == self.token.lower() else "upper"
        self.last_char = self.token[-1]
        self.geoText_loc = bool(GeoText(self.token).cities or GeoText(self.token).countries or (self.token in countries_list))
        self.known_org = self.token in orgs_list
        self.known_prog = self.token in progs_list
        self.hyphen = "-" in self.token

        self.prev_token = None # these can only be filled in from Sentence object
        self.next_token = None
        self.pos = None
        self.chunktag = None
        self.next_pos = None
        self.prev_pos = None
        self.next_chunktag = None
        self.prev_chunktag = None

    @classmethod
    def load_from_json(self, json_filepath):
        pass


class Sentence(object):
    """
    Sentence is a container for a list of Word objects
    """
    def __init__(self, words_objects, list_of_words):
        self.words_objects = words_objects # List of Word objects
        self.list_of_words = list_of_words # List of words in the sentence


class POSTaggedSentence(Sentence):
    """
    Like a Sentence, but with an attribute for POS tags.
    """
    def __init__(self, sentence, pos):

        super().__init__(sentence.words_objects, sentence.list_of_words)

        self.pos = pos

    @classmethod
    def load_from_json(cls, sentence_json_filepath):
        """
        Loads each Sentence JSON that was written to file by posTagger
        """
        sentence_json = json.load(open(sentence_json_filepath))

        words_objects = [Token(w["token"], w["sentence_position"], w["sentence_start"], w["sentence_end"], w["NEtag"])
            for w in sentence_json["sentence_words"]
            ]

        loaded_sentence = Sentence(words_objects, sentence_json["sentence_LOW"])

        return cls(loaded_sentence, sentence_json["sentence_POS"])


class ChunkTaggedSentence(Sentence):
    """
    Like a posTaggedSentence, but with chunk tags
    """

    def __init__(self, posTaggedSentence, chunk_tags):

        super().__init__(posTaggedSentence.words_objects, posTaggedSentence.list_of_words)
        self.pos = posTaggedSentence.pos
        self.chunk_tags = chunk_tags

    @classmethod
    def load_from_json(cls, sentence_json_filepath):
        """
        Loads each Sentence JSON that was written to file by chunkTagger
        """
        sentence_json = json.load(open(sentence_json_filepath))

        words_objects = [
            Token(w["token"], w["sentence_position"], w["sentence_start"], w["sentence_end"], w["NEtag"])
            for w in sentence_json["sentence_words"]
            ]

        loaded_sentence = Sentence(words_objects, sentence_json["sentence_LOW"])

        loaded_pos_sentence = POSTaggedSentence(loaded_sentence, sentence_json["sentence_POS"])

        return cls(loaded_pos_sentence, sentence_json["chunk_tags"])


class saturatedSentence(Sentence):

    def __init__(self, chunkTaggedSentence, countries_list, orgs_list, progs_list):
        super().__init__(chunkTaggedSentence.words_objects, chunkTaggedSentence.list_of_words)

        self.pos = chunkTaggedSentence.pos
        self.chunk_tags = chunkTaggedSentence.chunk_tags

        self.saturateTokens(countries_list, orgs_list, progs_list) # convert Tokens to saturatedTokens
        self.add_neighbor_token_features()
        self.add_pos_and_chunktag() # adding tags to each saturatedToken object
        self.add_neighbor_pos_and_chunktag()


    def saturateTokens(self, countries_list, orgs_list, progs_list):
        """
        Make each Token in words_objects a saturatedToken
        """
        self.words_objects = [
            saturatedToken(word_obj, countries_list, orgs_list, progs_list)
            for word_obj in self.words_objects
            ]

    def add_neighbor_token_features(self):
        neighbor_tokens = [w.token for w in self.words_objects]
        neighbor_tokens.insert(0, None) # for "prev" token of first word in sentence
        neighbor_tokens.append(None) # for "next" token of last word in sentence

        for index, word in enumerate(self.words_objects):
            word.prev_token = neighbor_tokens[index]
            word.next_token = neighbor_tokens[index + 2]


    def add_pos_and_chunktag(self):
        for index, w in enumerate(self.words_objects):
            p, c = self.chunk_tags[index]
            w.pos = p
            w.chunktag = c

    def add_neighbor_pos_and_chunktag(self):
        pos_seq = [w.pos for w in self.words_objects]
        chunk_seq = [w.chunktag for w in self.words_objects]

        pos_seq.insert(0, None)
        pos_seq.append(None)
        chunk_seq.insert(0, None)
        chunk_seq.append(None)


        for index, word in enumerate(self.words_objects):
            word.prev_pos = pos_seq[index]
            word.next_pos = pos_seq[index + 2]
            word.prev_chunktag = chunk_seq[index]
            word.next_chunktag = chunk_seq[index + 2]



