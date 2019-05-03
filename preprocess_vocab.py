import sys, re, string
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
from collections import Counter

# data
loans = pd.read_csv("data/kiva_loans.csv")
# columns that are prior knowledge
# loans = loans.loc[:, ["id", "loan_amount", "activity", "sector", "use",
#                      "country_code", "tags", "borrower_genders"]]
uses = loans.loc[:, "use"].dropna()
tags = loans.loc[:, "tags"].dropna()

# text processing stuff 
punc_regex = re.compile("[{}]".format(re.escape(string.punctuation)))
stop_words = set(stopwords.words('english'))

vocab = Counter()
min_threshold = 2

# progress tracking
processed = 0
total = len(uses.index) + len(tags.index)
update_interval = total // 20

def build_vocab(text):
    text = punc_regex.sub("", text)
    tokens = text.lower().split()
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 1]

    global vocab
    vocab.update(tokens)

    global processed
    processed += 1
    print_progress(processed, total)

    return tokens

def print_progress(processed, total):
    if processed % update_interval == 0:
        sys.stdout.write("Processed {} rows out of {} ({:.0f}%).\r".format(processed, total, processed / total * 100))
        sys.stdout.flush()

def main(argv):
    uses.apply(build_vocab)
    tags.apply(build_vocab)

    global vocab
    for k, v in list(vocab.items()):
        if v < min_threshold:
            del vocab[k]

    with open("vocab.p", 'wb') as vocab_pickle:
        pickle.dump(vocab, vocab_pickle)

    sys.stdout.write("\nWrote vocab counter to vocab.p\n")

if __name__ == "__main__":
    main(sys.argv)

