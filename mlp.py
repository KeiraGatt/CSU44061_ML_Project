#Paddy Flanagan
#MLP classifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def main():

    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    features = df.drop(['chargeable'], axis=1)
    target = df['chargeable']


    #split data into training and testing
    Xtrain, Xtest, ytrain, ytest = train_test_split(features,target,test_size=0.2)

    #use SMOTE k=21 on training data
    sm = SMOTE(sampling_strategy='auto', k_neighbors=21)
    X_res, y_res = sm.fit_resample(Xtrain, ytrain)
    df_train=pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)
    ytrain = df_train['chargeable']
    Xtrain = df_train.drop(['chargeable'], axis=1)


    #cross validation to choose the size of the hidden layers
    crossval=False
    if crossval:
        mean_error=[]
        std_error=[]
        hidden_layer_range= [2,5,10,15,25,50,75,100]
        hidden_layer_range= [60,70,75,80,90]
        for n in hidden_layer_range:
            print('hidden layer size : ', n,'\n')
            model = MLPClassifier(hidden_layer_sizes=(n), max_iter=300)
            scores = cross_val_score(model, Xtrain, ytrain, cv=5, scoring='f1')
            mean_error.append(np.array(scores).mean())
            std_error.append(np.array(scores).std())

        plt.errorbar(hidden_layer_range, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel('#hidden layer nodes')
        plt.ylabel('F1')
        plt.show()

        print(mean_error)

        #size of 75 seems best, after this it decreases

    # cross val to choose L2 regularization term 
    cross_val=False
    if cross_val:
        mean_error=[]
        std_error=[]
        C_range=[10000,1000,100,10,1,0.1,0.01,0.001]
        for Ci in C_range:
            print("C %d\n"%Ci)
            model=MLPClassifier(hidden_layer_sizes=(75), alpha = 1.0/Ci)
            scores = cross_val_score(model, Xtrain, ytrain, cv=5, scoring='f1')
            mean_error.append(np.array(scores).mean())
            std_error.append(np.array(scores).std())

        plt.errorbar(C_range, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel('C')
        plt.ylabel('F1')
        plt.show()

        print(mean_error)

        #1/10000 is best it seems



    #####testing against dummy model
    ##accuracies, f1 score, confusion matrix

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    model=MLPClassifier(hidden_layer_sizes=(75), alpha = 0.0001)
    model.fit(Xtrain, ytrain)
    preds = model.predict(Xtest)
    ydummy = dummy.predict(Xtest)

    print('Neural Net accuracy: ',accuracy_score(ytest, preds))
    print('dummy accuracy: ',accuracy_score(ytest, ydummy)) 


    print('F1 score of Neural Net: ', f1_score(ytest, preds))
    print('F1 score of dummy: ', f1_score(ytest, ydummy))

    #confusion matrices
    ConfusionMatrixDisplay.from_predictions(ytest, preds)
    plt.show()
    ConfusionMatrixDisplay.from_predictions(ytest, ydummy)
    plt.show()


    #AUC
    dummy_auc = roc_auc_score(ytest, ydummy)
    mlp_auc = roc_auc_score(ytest, preds)
    print('Dummy: ROC AUC=%.3f' % (dummy_auc))
    print('Neural Net: ROC AUC=%.3f' % (mlp_auc))

    dummy_fpr, dummy_tpr, _ = roc_curve(ytest, ydummy)
    mlp_fpr, mlp_tpr, _ = roc_curve(ytest, preds)

    pyplot.plot(dummy_fpr, dummy_tpr, linestyle='--', label='dummy')
    pyplot.plot(mlp_fpr, mlp_tpr, marker='.', label='Neural Net')

    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate') 

    pyplot.legend()
    pyplot.show()

  

if __name__ == '__main__':
    main()

