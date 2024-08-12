from libraries import *
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,confusion_matrix

def print_score(test, pred, model):
    fig, ax = plt.subplots(1,2,figsize=(15, 6))
    sns.heatmap(confusion_matrix(test, pred), annot=True, cmap='Blues', ax=ax[0], center=True, linecolor='black', square=True)
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    ax = sns.distplot(test, color='r',  label='Actual Value',hist=False)
    sns.distplot(pred, color='b', label='Predicted Value',hist=False,ax=ax)
    plt.title(f'Actual vs Predicted Value {model}', fontweight = 'bold')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.show()
    Metrics = {'Metrics':['Accuracy Score', 'f1 Score', 'Mean Absolute Error ','Mean Squared Error', 'R2 Score'],
               'Score' : [accuracy_score(test, pred), f1_score(test, pred, average="weighted"),
                          mean_absolute_error(test, pred),mean_squared_error(test, pred),r2_score(test, pred)]}
    df = pd.DataFrame(Metrics)
    return df

# models = ['DecisionTree','SVM','KNN']
# preds = [dtc_pred,svm_pred,knn_pred]
# accuracys= []
# for i in preds:
#     accuracys.append( accuracy_score(y_test, i))
# plt.figure(figsize=(10,6), dpi = 300)
# sns.barplot(x=models, y=accuracys)
# plt.xlabel('Algorithms')
# plt.ylabel('Accuracy Score')
# plt.title('Comparison of different models', fontweight = 'bold')