from libraries import *

for dirname, _, filenames in os.walk('C:/Users/Farhan Nafis/Documents/jupitercodes/AI_Project/kaggle'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

heart_df = pd.read_csv("C:/Users/Farhan Nafis/Documents/jupitercodes/AI_Project/kaggle\heart.csv")