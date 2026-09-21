import pandas as pd
import numpy as np

df = pd.read_csv('raw_data/Loan_default.csv')
#print(df.head(5))
print(df.columns)
print(df.info())
#print(df.isnull().sum())
print(df.dropna())
print(df.duplicated().sum())