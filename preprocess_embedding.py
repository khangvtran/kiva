import sys, pickle, re, string
import numpy as np
import pandas as pd
from collections import Counter
from keras.preprocessing.text import Tokenizer

# text processing stuff
punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))

# load vocab data
with open('vocab.p', 'rb') as vocab_pickle:
    vocab = pickle.load(vocab_pickle)
vocab_set = set(vocab.keys())

# data
with open('X_train.p', 'rb') as X_train_pickle:
    X_train = pickle.load(X_train_pickle)
uses_train = X_train.loc[:, "use"]
tags_train = X_train.loc[:, "tags"]

# progress tracking
processed = 0
total = len(uses.index) + len(tags.index)
update_interval = total // 20

def print_progress(processed, total):
    if processed % update_interval == 0:
        sys.stdout.write("Processed {} rows out of {} ({:.0f}%)\r".format(processed, total, processed / total * 100))
        sys.stdout.flush()

def tokenize(text)
    """Clean text string by transforming into a space-separated string
    of words in vocab
    """
    text = punc_regex.sub("", text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t in vocab]
    tokens = " ".join(tokens)

    global processed
    processed += 1
    print_progress(processed, total)

    return tokens

def fit_tokenizer():
    sys.stdout.write("\nTokenizing text data...")
    sys.stdout.flush()

    uses_tokens = uses_train.map(tokenize)
    tags_tokens = tags_train.map(tokenize)
    tokens = uses_tokens + " " + tags_tokens

    sys.stdout.write("\nFitting tokenizer on all tokens...")
    sys.stdout.flush()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)

    sys.stdout.write(" Done\n")
    sys.stdout.flush()

    # find max length token string for padding
    max_len = tokens.map(len).max()

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
