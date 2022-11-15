import pandas as pd
import os

description = pd.read_csv("data/company_reviews.csv")
description = description.drop_duplicates("name")
description = description.dropna(subset="name")
description = description[["name", "description"]]
description["name"] = description["name"].str.lower()

industry = pd.read_csv("data/companies_sorted.csv")
industry = industry[["name", "industry"]]

data = pd.merge(description, industry, how='left', on='name')
data = data.drop_duplicates("name")
data = data[["description", "industry"]]
data = data.dropna()

os.remove("data/company_reviews.csv")
os.remove("data/companies_sorted.csv")

data.to_csv("data//data.csv", index=False)
