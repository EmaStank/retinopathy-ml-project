from scipy.io import arff
import pandas as pd

data, meta = arff.loadarff("messidor_features.arff")
df = pd.DataFrame(data)

for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode("utf-8")

print("First rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))

print("\nInfo:")
print(df.info())

print("\nBasic statistics:")
print(df.describe())