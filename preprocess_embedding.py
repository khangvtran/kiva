import sys, pickle, re, string
import numpy as np
import pandas as pd
from collections import Counter

# text processing stuff
punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))

# load vocab data
with open('vocab.p', 'rb') as vocab_pickle:
    vocab = pickle.load(vocab_pickle)
vocab_set = set(vocab.keys())

loans = pd.read_csv("data/kiva_loans.csv")
uses = loans.loc[:, "use"].dropna()
tags = loans.loc[:, "tags"].dropna()

# progress tracking
processed = 0
total = len(uses.index) + len(tags.index)
update_interval = total // 20

def print_progress(processed, total):
    if processed % update_interval == 0:
        sys.stdout.write("Processed {} rows out of {} ({:.0f}%).\r".format(processed, total, processed / total * 100))
        sys.stdout.flush()

def clean_text(text)
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

def main(argv):
    pass

if __name__ == "__main__":
    main(sys.argv)
