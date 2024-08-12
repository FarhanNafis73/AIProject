from encoding import *
from libraries import *

class Pie:
    def __init__(self, df):
        self.df = df    

    def pieplot(self):
        myexplode = [0.1, 0]
        plt.figure(figsize=(10,8))
        plt.pie(df['HeartDisease'].value_counts(), labels=['Yes','No'], explode = myexplode, startangle = 90, shadow = True, 
                colors = ['red', 'green'], autopct='%1.1f%%')
        plt.title('Heart Disease Percentage', fontweight='bold')
        plt.show()

class Count:
    def __init__(self, df):
        self.df = df

    def cntplot(self):
        plt.figure(figsize=(10,8))
        sns.countplot(x='Sex', data=df, hue='HeartDisease')
        plt.title('Gender Distribution', fontweight = 'bold')
        plt.xlabel('Gender - [M : 0 | F : 1]')
        plt.ylabel('Count')
        plt.show()

class Scat:
    def __init__(self, df):
        self.df = df

    def sctplot(self):
        plt.figure(figsize=(10,8))

        plt.scatter(df.Age[df.HeartDisease==1],
        df.MaxHR[df.HeartDisease==1],
        c="salmon")

        plt.scatter(df.Age[df.HeartDisease==0],
        df.MaxHR[df.HeartDisease==0],
        c="lightblue");

        plt.title("Heart disease in function of Age and Max heart Rate", fontweight = 'bold')
        plt.xlabel("Age")
        plt.ylabel("Max Heart Rate")
        plt.legend(["Disease","No Disease"])
        plt.show()

class Group:
    def __init__(self, df):
        self.df = df
    
    def grph(self):
        fig,ax = plt.subplots(2,3,figsize=(20,10))
        sns.histplot(x = 'RestingBP', data = df, ax = ax[0,0], bins = 10)
        sns.histplot(x = 'Cholesterol', data = df, ax = ax[0,1], bins = 10)
        sns.countplot(x = 'RestingECG', data = df, ax = ax[0,2])
        sns.histplot(x = 'MaxHR', data = df, ax = ax[1,0], bins = 10)
        sns.countplot(x = 'ExerciseAngina', data = df, ax = ax[1,1]) 
        sns.countplot(x = 'ST_Slope', data = df, ax = ax[1,2])
        plt.show()

class Correlation:
    def __init__(self, df):
        self.df = df

    def corplot(self):
        df.corr()['HeartDisease'][:-1].sort_values().plot(kind='bar', figsize=(20,10), color = 'salmon')
        plt.show()

class Agegroup:
    def __init__(self, df):
        self.df = df

    def agegroupplot(self):
        plt.figure(figsize=(10,6))
        df.Age.plot.hist()
        plt.show()

class Heatmap:
    def __init__(self, df):
        self.df = df
    
    def heatplot(self):
        plt.figure(figsize=(20,15))
        sns.heatmap(df.corr(), annot=True, linewidths=0.5, fmt=".2f", cmap="YlGnBu", center=True, square=True, linecolor='black')
        plt.show()

class Angina:
    def __init__(self, df):
        self.df = df

    def angiplot(self):
        plt.figure(figsize=(10,6))
        sns.countplot(x='ExerciseAngina', data=df, hue='HeartDisease')
        plt.title('ExerciseAngina Distribution', fontweight = 'bold')
        plt.xlabel('ExerciseAngina - [No : 0 | Yes : 1]')
        plt.ylabel('Count')
        plt.show()

class AgeHeartD:
    def __init__(self, df):
        self.df = df

    def ageplot(self):
        plt.figure(figsize=(10,6))
        sns.histplot(data=df, x="Age", hue="HeartDisease", multiple="stack",kde=True)
        plt.show()
        
