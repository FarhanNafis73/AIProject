from dataset import *
from sklearn.preprocessing import LabelEncoder

print("#############################################################")
cat_col = []
for col in df.select_dtypes(include= 'object' ).columns:
    if df[col].nunique() < 8:
        print(df[col].value_counts())
        print('-'*40)
        cat_col.append(col)
print("#############################################################")
print("\n")

print(df['Cholesterol'].value_counts())
print("\n")

for col in cat_col:
    print(col)
    print(df[col].unique(), list(range(df[col].nunique())))
    
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])
    
    print('-' * 40)

from sklearn.impute import KNNImputer

df['Cholesterol'].replace(0, np.nan, inplace=True)
imputer = KNNImputer(n_neighbors=3)
after_impute = imputer.fit_transform(df)

df = pd. DataFrame(after_impute, columns=heart_df.columns)

selected_Col = df.columns
selected_Col = selected_Col.drop('Oldpeak')
df[selected_Col] = df[selected_Col].astype('int32')