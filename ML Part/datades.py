from dataset import *

print(heart_df.head(5))
print("\n")
print(heart_df.tail(5))
print("\n")
print(df.info())
print("\n")
print(df.describe())
print("\n")
df.isnull().sum()
print("\n")
df.duplicated().sum()
print("\n")
df.nunique().sort_values(ascending=False)
print("\n")