from encoding import *
from sklearn.model_selection import train_test_split 

#Test size = 0.2 = 20%. For train we are keeping 80%.
X_train, X_test, y_train, y_test = train_test_split(df.drop('HeartDisease', axis=1), df['HeartDisease'], test_size=0.2, random_state=42)