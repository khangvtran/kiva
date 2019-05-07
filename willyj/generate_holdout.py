import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

print("Loading data...")
loans = pd.read_csv('../data/kiva_loans.csv')

# take columns that are prior knowledge and drop NaNs
prior_cols = ['id', 'loan_amount', 'activity', 'sector', 'use',
              'country_code', 'tags', 'borrower_genders']
loans = loans.loc[:, prior_cols]

print("Size before dropping NAs: {}".format(len(loans.index)))
loans = loans.dropna()
print("Size after dropping NAs: {}".format(len(loans.index)))

loan_amount = loans.loc[:, 'loan_amount']
loans.drop('loan_amount', axis=1, inplace=True)

test_size = 0.2
train_size = 1 - test_size
subsample=0.2

print("Splitting {:.0f}% of data ({} rows)into train ({:.0f}%) and test ({:.0f}%)..."\
        .format(subsample * 100, subsample * len(loan_amount.index), train_size * 100, test_size * 100))

X_train, X_test, y_train, y_test = train_test_split(loans, loan_amount,
                                    test_size=test_size * subsample,
                                    train_size=train_size * subsample,
                                    random_state=42)

with open("pickles/X_train.pickle", 'wb') as X_train_pickle:
    pickle.dump(X_train, X_train_pickle)
    print("Wrote X_train to pickles/X_train.pickle")

with open("pickles/X_test.pickle", 'wb') as X_test_pickle:
    pickle.dump(X_test, X_test_pickle)
    print("Wrote X_test to pickles/X_test.pickle")

with open("pickles/y_train.pickle", 'wb') as y_train_pickle:
    pickle.dump(y_train, y_train_pickle)
    print("Wrote y_train to pickles/y_train.pickle")

with open("pickles/y_test.pickle", 'wb') as y_test_pickle:
    pickle.dump(y_test, y_test_pickle)
    print("Wrote y_test to pickles/y_test.pickle")
