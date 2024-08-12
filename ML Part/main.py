from split import *
from results import *
import time

def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV

    dtree = DecisionTreeClassifier(class_weight='balanced')
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3, 4],
        'random_state': [0, 42]
    }
    grid_search = GridSearchCV(dtree, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    Ctree = DecisionTreeClassifier(**grid_search.best_params_, class_weight='balanced')
    Ctree.fit(X_train, y_train)
    dtc_pred = Ctree.predict(X_test)
    print("Accuracy from the Decision Tree Algorithm: ", accuracy_score(y_test, dtc_pred))
    print_score(y_test, dtc_pred,'Decision Tree Classifier')

def svm():
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    kernels = {'linear':0, 'poly':0, 'rbf':0, 'sigmoid':0}
    best = ''
    for i in kernels:
        svm = SVC(kernel=i)
        svm.fit(X_train, y_train) 
        yhat = svm.predict(X_test)
        kernels[i]=f1_score(y_test, yhat, average="weighted")
        if kernels[i] == max(kernels.values()):
            best = i
    svm = SVC(kernel=best)
    svm.fit(X_train, y_train) 
    svm_pred = svm.predict(X_test)
    print(f'SVM algorithm F1_score kernel ({best}): {f1_score(y_test, svm_pred, average="weighted")}')
    print_score(y_test, svm_pred, 'Support Vector Machine')

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    Ks = 50
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
        
        neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=neigh.predict(X_test)
        mean_acc[n-1] = accuracy_score(y_test, yhat)
        std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

    best_K = mean_acc.argmax()+1
    knn = KNeighborsClassifier(n_neighbors = best_K).fit(X_train,y_train)
    knn_pred=neigh.predict(X_test)
    print( "The best accuracy from KNN was ", accuracy_score(y_test, knn_pred))
    print_score(y_test, knn_pred, 'KNN') 

if __name__ == "__main__":
    t0 = time.time()
    print(f"Training Started")
    svm()
    print(f"Training Finished")
    t1 = time.time()
    print(f"Training Time: {1000*(t1-t0)} ms")

