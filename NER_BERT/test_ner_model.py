import os
import itertools
import pandas as pd
import numpy as np
import wandb
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch

label_list = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
label_encoding_dict = {'I-PRG': 2, 'I-I-MISC': 2, 'I-OR': 6, 'O': 0, 'I-': 0, 'VMISC': 0, 'B-PER': 3, 'I-PER': 4,
                       'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8, 'B-MISC': 1, 'I-MISC': 2}

task = "ner"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

Train = False

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



def get_all_tokens_and_ner_tags(directory):
    return pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in
                      os.listdir(directory)]).reset_index().drop('index', axis=1)


def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})


def get_un_token_dataset(train_directory, test_directory):
    train_df = get_all_tokens_and_ner_tags(train_directory)
    test_df = get_all_tokens_and_ner_tags(test_directory)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return (train_dataset, test_dataset)


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        # return self.tokenized_texts.numpy().size
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

def ner_detect(sentence):

    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=test_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    wandb.login(key='168480f158655b007fd002fbc7f5639afc3c2126')

    if Train == True:
        trainer.train()
        trainer.evaluate()
        trainer.save_model('un-ner.model')

    test_encodings = tokenizer([sentence],
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")  # .input_ids

    test_dataset = SimpleDataset(test_encodings)
    res = trainer.predict(test_dataset)

    label_to_id = {l: i for i, l in enumerate(label_list)}
    id_to_label = {label_to_id[x]: x for x in label_to_id}
    decoded = []
    words = []
    FINAL = []
    for i, r in enumerate(res[0][0]):
        # res[0][0][i].argmax()
        decoded.append(id_to_label[r.argmax()])
        words.append(tokenizer.decode(test_encodings.data['input_ids'][0][i]))
        if id_to_label[r.argmax()] != 'O' and tokenizer.decode(test_encodings.data['input_ids'][0][i]) != '[PAD]':
            FINAL.append((tokenizer.decode(test_encodings.data['input_ids'][0][i]), id_to_label[r.argmax()]))

    print(FINAL)
    return FINAL
if __name__ == "__main__":
    train_dataset, test_dataset = get_un_token_dataset('./UN-named-entity-recognition/tagged-training/',
                                                       './UN-named-entity-recognition/tagged-test/')

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

    # model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))
    model = AutoModelForTokenClassification.from_pretrained(
        "/Users/jordanharris/SCAPT-ABSA/NER_BERT/test-un-ner/checkpoint-500/",
        num_labels=len(label_list))

    model = model.from_pretrained("/Users/jordanharris/SCAPT-ABSA/NER_BERT/un-ner.model")

    args = TrainingArguments(
        f"test-{task}",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=1e-5,
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")


 #--------------------------------------------------------------------------------------------------------
    sentence = "Before proceeding further, I should like to inform members that action on draft resolution iv, entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date to allow time for the review of its programme budget implications by the fifth committee. The assembly will take action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or adoption."
    ner_detect(sentence)



 # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
    # texts = ["Hello there!", "This is another text"]
    # tokenized_texts = tokenizer(texts, padding=True)
    #
    #
    # class SimpleDataset:
    #     def __init__(self, tokenized_texts):
    #         self.tokenized_texts = tokenized_texts
    #
    #     def __len__(self):
    #         return len(self.tokenized_texts["input_ids"])
    #
    #     def __getitem__(self, idx):
    #         return {k: v[idx] for k, v in self.tokenized_texts.items()}
    #
    #
    # test_dataset = SimpleDataset(tokenized_texts)
    # trainer = Trainer(model=model)
    # predictions = trainer.predict(test_dataset)
    # predictions.predictions
