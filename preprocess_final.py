import pandas as pd

loans = pd.read_csv("data/kiva_loans.csv")

# columns that are prior knowledge
loans = loans.loc[:, ["id", "loan_amount", "activity", "sector", "use",
                      "country_code", "tags", "borrower_genders"]]
