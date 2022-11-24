#Aaron Readman
#Knn classifier model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



#Function Definition: Create Various plots to inform choice of K Value 
#Plot F1 scores against  different K values with cross validation
#inputs are array of k values, training data and training test data
def chooseKValue(k_range,X,y):
    print("Choosing k Value")
    
    mean_error=[]; std_error=[]
    for k in k_range:

        model = KNeighborsClassifier(n_neighbors=k,weights="uniform")
        scores = cross_val_score(model, X, y, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    plt.errorbar(k_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel("K")
    plt.ylabel("F1 Score")
    plt.title("Knn Cross Validation Versus F1 Score")
    plt.show()


def main():
        
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)

    #Our Test and Training Data
    Xtrain, Xtest, ytrain, ytest = train_test_split(features,target,test_size=0.2)

    #########################
    #Cross Validation for KNN
    Ki_range= [1, 5, 10, 15, 20, 25, 50, 100]
    chooseKValue(Ki_range,Xtrain,ytrain)

    Ki_range= [2,4,5,6,8,10]
    chooseKValue(Ki_range,Xtrain,ytrain)
    #########################

    ##############Model Results
    #Knn
    #K values seem to alternate between 5-10 as the ideal choice
    modelKnn = KNeighborsClassifier(n_neighbors=10,weights="uniform").fit(Xtrain, ytrain)
    yPredictions = modelKnn.predict(Xtest)

    print("Knn Information Data Report")
    print(confusion_matrix(ytest, yPredictions))
    print(classification_report(ytest, yPredictions))

    #Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(ytest, yPredictions).ravel()
    print(tn, fp, fn, tp)

    #Printing the Accuracy Score
    print(accuracy_score(ytest, yPredictions))


    #Plotting the ROC curve
    plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
    xProbability=modelKnn.predict_proba(Xtest)


    #Plotting Values
    fpr, tpr, _ = roc_curve(ytest,xProbability[:, 1])
    plt.plot(fpr,tpr, label="modelKnn",c="b")


    #Plotting details
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')

    plt.title("ROC Curve")
    plt.plot([0, 1], [0, 1], color='red',linestyle='--')
    plt.legend(framealpha=1, loc="lower right" )
    plt.show()


if __name__ == '__main__':
    main()



