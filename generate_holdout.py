import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

print("Loading data...")
loans = pd.read_csv('data/kiva_loans.csv')

# columns that are prior knowledge
prior_cols = ['id', 'loan_amount', 'activity', 'sector', 'use',
              'country_code', 'tags', 'borrower_genders']
loans = loans.loc[:, prior_cols].dropna()
loan_amount = loans.loc[:, 'loan_amount']
loans.drop('loan_amount', axis=1, inplace=True)

test_size = 0.2
train_size = 1 - test_size

print("Splitting data into train ({:.0f}%) and test ({:.0f}%)...".format(train_size * 100, test_size * 100))
X_train, X_test, y_train, y_test = train_test_split(
        loans, loan_amount, test_size=test_size, random_state=42)

with open("X_train.p", 'wb') as X_train_pickle:
    pickle.dump(X_train, X_train_pickle)
    print("Wrote X_train to X_train.p")

with open("X_test.p", 'wb') as X_test_pickle:
    pickle.dump(X_test, X_test_pickle)
    print("Wrote X_test to X_test.p")

with open("y_train.p", 'wb') as y_train_pickle:
    pickle.dump(y_train, y_train_pickle)
    print("Wrote y_train to y_train.p")

with open("y_test.p", 'wb') as y_test_pickle:
    pickle.dump(y_test, y_test_pickle)
    print("Wrote y_test to y_test.p")
