#!/usr/bin/env python3

# Leslie Huang
# Pipeline Part 2: Chunking
# This script imports posTaggedSentence objects,
# chunk tags them,
# and outputs chunkTaggedSentence objects

from sentences import Sentence, Token, POSTaggedSentence, ChunkTaggedSentence
from taggers import UnigramChunker, POSTagger

import argparse
import itertools
import json
from multiprocessing import Pool
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("posTagged_JSON_dir", help = "dir to read posTagged Sentence object JSON files from")
parser.add_argument("manually_chunked_JSON", help = "dir of JSONs of manually tagged chunk tags")
parser.add_argument("output_dir", help = "dir to write chunkTaggedSentence JSONs to")
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)


if __name__ == '__main__':

    pool = Pool(os.cpu_count() - 1)

    # import pos tagged data and convert to list of posTaggedSentence objects
    imported_pos_tagged_sentences = []

    for fn in os.listdir(args.posTagged_JSON_dir):

        imported_pos_tagged_sentences.append(
            POSTaggedSentence.load_from_json((os.path.join(args.posTagged_JSON_dir, fn))
                )

        )

    # Initialize chunker
    chunker = UnigramChunker.read_from_json(args.manually_chunked_JSON)
    # chunker = UnigramChunker.load_nltk_chunked_sentences() # CONLL trained chunker

    for index, result in enumerate(pool.imap(chunker.tag_sentences, imported_pos_tagged_sentences)):

        s = ChunkTaggedSentence(imported_pos_tagged_sentences[index], result)

        out_dict = {
                "sentence_LOW": s.list_of_words,

                "sentence_words": [{
                    "token": word.token,
                    "sentence_position": word.sentence_position,
                    "sentence_start": word.sentence_start,
                    "sentence_end": word.sentence_end,
                    "NEtag": word.NEtag
                }

                     for word in s.words_objects],

                "chunk_tags": result,
                "sentence_POS": s.pos,
                }

        with open(os.path.join(args.output_dir, f'{index}.json'), 'w') as f:
            json.dump(out_dict, f)


    # # Uncomment this block to output formatted word,pos,chunk data for manual chunk tagging
    # out = []
    # for sentence in imported_pos_tagged_sentences:
    #     out.append(chunker.tag_sentences(sentence))
    # with open("chunks_manual.json", "w") as f:
    #     json.dump(out, f)