import sys, pickle, re, string
import numpy as np
import pandas as pd
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# text processing stuff
punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))

# load the data
with open('vocab.pickle', 'rb') as vocab_pickle:
    vocab = pickle.load(vocab_pickle)
vocab_set = set(vocab.keys())

with open('X_train.pickle', 'rb') as X_train_pickle:
    X_train = pickle.load(X_train_pickle)
uses_train = X_train.loc[:, "use"]
tags_train = X_train.loc[:, "tags"]
X_train_texts = [uses_train, tags_train]

with open('X_test.pickle', 'rb') as X_test_pickle:
    X_test = pickle.load(X_test_pickle)
uses_test = X_test.loc[:, "use"]
tags_test = X_test.loc[:, "tags"]
X_test_texts = [uses_test, tags_test]

with open('y_train.pickle', 'rb') as y_train_pickle:
    y_train = pickle.load(y_train_pickle)

with open('y_test.pickle', 'rb') as y_test_pickle:
    y_test = pickle.load(y_test_pickle)

# progress tracking
processed = 0
total = sum([len(t.index) for t in X_train_texts])
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

def fit_tokenizer(texts):
    """Return a tokenizer model fitted to the texts in the input list,
    as well as the converted encodings. The texts in each input series
    will be concatenated together separated by spaces.

    Parameters
    ----------
    texts: list
        list of pandas series containing rows of text to encode
    """
    # tokenize the stuff
    sys.stdout.write("\nTokenizing text data...")
    sys.stdout.flush()

    global processed
    processed = 0

    tokens = texts[0].map(tokenize)
    for t in texts[1:]:
        tokens = tokens + " " + t.map(tokenize)

    # fit the tokenizer
    sys.stdout.write("\nFitting tokenizer on all tokens...")
    sys.stdout.flush()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)

    sys.stdout.write(" Done\n")
    sys.stdout.flush()

    # find max length token string for padding
    maxlen = tokens.map(len).max()

    # convert tokens to encoded sequences
    sys.stdout.write("\nConverting tokens to numerical sequences...")
    sys.stdout.flush()

    sequences = tokenizer.text_to_sequences(tokens)

    sys.stdout.write(" Done\n")
    sys.stdout.flush()

    # pad the sequences
    sys.stdout.write("\nConverting tokens to numerical sequences...")
    sys.stdout.flush()

    padded_sequences = pad_sequences(sequences, maxlen=maxlen, padding='post')

    sys.stdout.write(" Done\n")
    sys.stdout.flush()

    return tokenizer, padded_sequences

def train_nn(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
 
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit network
    model.fit(X_train, y_train, epochs=10, verbose=2)

    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: {}'.format(acc * 100)) 

def main(argv):
    train_tokenizer, train_sequences = fit_tokenizer(X_train_texts)
    test_tokenizer, test_sequences = fit_tokenizer(X_test_texts)
    train_nn(train_sequences, test_sequences, y_train, y_test)

if __name__ == "__main__":
    main(sys.argv)
