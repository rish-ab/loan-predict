import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#print(df.head(5))
print(df.columns)
print(df.info())
#print(df.isnull().sum())
df = df.dropna()
print(df.head(10))