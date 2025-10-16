import pandas as pd
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

df = pd.read_excel(url, header=1)

os.makedirs("data", exist_ok=True)

df.to_csv("data/credit_card_default.csv", index=False)