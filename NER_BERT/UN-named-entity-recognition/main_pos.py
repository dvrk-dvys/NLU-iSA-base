#!/usr/bin/env python3

# Leslie Huang
# Pipeline Part 1: POS tagging

from sentences import RawDocument, Sentence, Token, POSTaggedSentence, ChunkTaggedSentence
from taggers import POSTagger

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help = "dir containing training or test data")
parser.add_argument("output_dir", help = "dir to write posTaggedSentence JSONs to")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# These patterns are taken from the NLTK Book, Chapter 5 http://www.nltk.org/book/ch05.html
re_expressions = patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
]


def load(dirname):
    """
    Load all text files from a directory and convert them to a list of Sentence objects
    """
    all_documents = []

    # load training texts into a collection of RawDocument objects
    for fn in os.listdir(dirname):
        if fn.endswith(".txt"):
            all_documents.append(
                RawDocument(True, os.path.join(dirname, fn))
            )

    container = []

    # Now convert each sentence into a Sentence of Words
    for document in all_documents:
        for sentence in document.text_grouped_into_sentences:
            words = []
            list_of_words = [i[0] for i in sentence]

            for counter, value in enumerate(sentence):
                    word, NEtag = value

                    words.append(
                        Token(word, counter, counter == 0, counter == len(sentence) - 1, NEtag)
                        )

            container.append(Sentence(words, list_of_words))

    return container


if __name__ == '__main__':

    input_data = load(args.input_dir) # This is a list of Sentences

    pool = Pool(os.cpu_count() - 1)


    tagger = POSTagger(re_expressions)


    for index, result in enumerate(pool.imap(tagger.tagger, input_data)):

        out_dict = {
                "sentence_LOW": input_data[index].list_of_words,

                "sentence_words": [{
                    "token": word.token,
                    "sentence_position": word.sentence_position,
                    "sentence_start": word.sentence_start,
                    "sentence_end": word.sentence_end,
                    "NEtag": word.NEtag
                }

                     for word in input_data[index].words_objects],

                "sentence_POS": result,
                }

        with open(os.path.join(args.output_dir, f'{index}.json'), 'w') as f:
            json.dump(out_dict, f)
