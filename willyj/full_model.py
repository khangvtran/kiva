import pandas as pd

loans = pd.read_csv("data/kiva_loans.csv")

# columns that are prior knowledge
# - use and tags are seemingly arbitrary so we use the word embeddings for those
# - activity and sector seem to be restricted to a reasonable set of values,
#   so we can just one-hot those
loans = loans.loc[:, ["id", "loan_amount", "activity", "sector", "use",
                      "country_code", "tags", "borrower_genders"]]
