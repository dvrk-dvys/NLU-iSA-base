# Award top K Hotels

import random
import spacy
from spacy.util import minibatch, compounding
from sklearn.model_selection import train_test_split


def load_training_data_choice(positiveKeywords, negativeKeywords):
    pos_key_words = positiveKeywords.split()
    neg_key_words = negativeKeywords.split()

    data = []
    train_choice = []
    test_choice = []

    # for index in range(len(x_train)):
    #     if y_train[index] == 'gratitude':
    #         x = (x_train[index], {'cats': {'gratitude': True, 'greeting': False, 'other': False}})
    #         train_ggo.append(x)


    for each in neg_key_words:
        x = (each, {'vals': {'pos': False, 'neg': True}})
        data.append(x)
    for every in pos_key_words:
        x = (every, {'vals': {'pos': True, 'neg': False}})
        data.append(x)

    X = []
    Y = []
    for each in data:
        X.append(each[0])
        Y.append(each[1])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

    train_zip = zip(X_train, y_train)
    train_list = list(train_zip)

    test_zip = zip(X_test, y_test)
    test_list = list(test_zip)

    return train_list, test_list



def train_model(
        training_data: list,
        test_data: list,
        iterations: int,
        data_directory: str
) -> None:
    # Build pipeline
    spacy.prefer_gpu()
    # config = spacy.util.load_config('./spacy.cfg')
    # nlp = spacy.load("en_core_web_sm", config=config)
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        nlp = spacy.blank("en")
        # nlp = spacy.blank("en", config=config)
        # textcat = nlp.create_pipe("textcat",
        #                           config={"architechture": "simple_cnn"}
        #                           )
        nlp.add_pipe("textcat", last=True)
    textcat = nlp.get_pipe("textcat")


    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            random.shuffle(training_data)

            losses = {}
            for batch in minibatch(training_data, size=batch_sizes):
                for text, annotations in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotations)
                    nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)

            # batches = minibatch(training_data, size=batch_sizes)
            # for batch in batches:
            #     text, labels = zip(*batch)
            #     nlp.update(
            #         text,
            #         labels,
            #         drop=0.2,
            #         sgd=optimizer,
            #         losses=loss
            #     )
            # with textcat.model.use_params(optimizer.averages):
            evaluation_results = evaluate_model(
                tokenizer=nlp.tokenizer,
                textcat=textcat,
                test_data=test_data
            )
            print(
                f"{losses['textcat']}\t{evaluation_results['precision']}"
                f"\t{evaluation_results['recall']}"
                f"\t{evaluation_results['f-score']}"
            )
    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk(data_directory)


def test_model(input_data, load):
    #  Load saved trained model
    loaded_model = spacy.load(load)
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]

    print(
        "_______________________________________________________________\n"
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}\n"
        "_______________________________________________________________\n"
    )
    return (prediction, score)


def evaluate_model(
        tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                    predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["vals"]["pos"]:
                true_positives += 3
            elif score >= 0.5 and true_label["vals"]["neg"]:
                false_positives = 0
            elif score < 0.5 and true_label["vals"]["neg"]:
                true_negatives -= 1
            elif score < 0.5 and true_label["vals"]["pos"]:
                false_negatives = 0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}



# functions description
# Complete the function awardTopKHotels in teh editor below.
# The function must return a list of hotel ids sorted in Desc order from total score



if __name__ == "__main__":
    pos = 'breakfast beach citycenter location metro view staff price'
    neg = 'not'

# # # Simple Binary Choice
    train, test = load_training_data_choice(pos, neg)
    train_model(train, test, 20, "hotel_model_artifacts")
    # test_model('Where is the bathroom', 'choice_model_artifacts_new')
