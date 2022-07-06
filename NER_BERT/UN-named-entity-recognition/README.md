# Named Entity Recognition system for political speech

## A full NLP pipeline

### About the data

The corpus consists of a sample of transcribed speeches given at the UN General Assembly from 1993-2016, which were scraped from the UN website, parsed (e.g. from PDF), and cleaned.

More than 50,000 tokens in the test data were manually tagged for Named Entity Recognition (O - Not a Named Entity; I-PER - Person; I-ORG - Organization; I-LOC - Location; I-MISC - Other Named Entity).

### How to run:

`pip install -r requirements.txt`

### 1 Part-of-Speech tagging (on `tagged-test/` and `tagged-training/`) using a sequence of NLTK taggers (Bigram, Unigram, Regex, Default)


`python3 main_pos.py input_dir output_dir`

Arguments:

- `input_dir` is a directory containing text files with a token and a Named Entity tag on each line separated by a tab (line of whitespace separates sentences)
- the script generates `output_dir` containing one JSON file per sentence in the `input_dir`
- Each JSON (file) is a dictionary of (1) a list of the words in the sentence as strings, (2) a list of [word, POS] pairs, and (3) a list of dictionaries, each of which represent a word and several features

### 2 Chunking using an NLTK Unigram Tagger and manually tagged training data

`python3 main_chunker.py input_dir training_chunk_data.json output_dir`

Arguments:

- `input_dir` is the output of the previous step
- `training_chunk_data.json` is manually chunktagged data
- the script generates `output_dir` containing JSON files that correspond to those in `input_dir`, with an additional key-value pair is added to represent a list of [word, pos, chunk tag]

### 3 Feature preparation for machine learning models
`python3 main_prep_for_ner.py input_dir output_dir`

Arguments:

- `input_dir` is the output of the previous step
- `output_dir` contains JSON files that correspond to those in `input_dir`, with additional features added to the word-dictionaries (see (3) in Step 1)

### 4 Ensemble Named Entity tagging with MaxEnt Markov Model, multiclass Naive Bayes, and Decision Tree Classifiers

`python3 main_ner.py training_dir test_dir num_iterations output_dir predicted_classifications.json true_classifications.json`

Arguments:

- `training_dir` is the output of the previous step for training data
- `test_dir` as above, for test data
- `num_iterations` for MaxEnt Classifier
-  `output_dir` contains one output file (technically invalid JSON) generated for each classifier, where each line is a word's dictionary of features and the predicted NER tag


### 5 Performance
`python3 scorer.py true_classifications.json predicted_classifications.json`

Arguments:
- true NER labels (JSON)
- output of the previous step (JSON)