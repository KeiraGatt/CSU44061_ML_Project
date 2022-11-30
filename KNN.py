#Aaron Readman
#Knn classifier model


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where

from sklearn.linear_model import LogisticRegression

import imblearn
from collections import Counter

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 ,SelectFpr


#Function to choose the K value 
#Will plot F1 score against K values using cross validation
#Takes a range of K values, feature vector X and classification data y
def chooseKValue(k_range,X,y,weightType):
    print("Choosing k Value")
    
    mean_error=[]; 
    std_error=[]

    #Iterating through k range
    for k in k_range:

        #Creating model
        model = KNeighborsClassifier(n_neighbors=k,weights=weightType)
        #Testing model
        scores = cross_val_score(model, X, y, cv=5, scoring="f1")
        #Getting mean and standard error
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    #Plotting K values against F1 score
    plt.errorbar(k_range,mean_error,yerr=std_error,linewidth=3)
    plt.xlabel("K")
    plt.ylabel("F1 Score")
    plt.title("Knn Cross Validation Versus F1 Score")
    plt.show()

#Function to rebalance training data
#Sampling Strategy set to 0.99 to ensure the dummy classification picks the correct class
def dataRebalance(Xtrain,ytrain):
    #############

    counter = Counter(ytrain)
    print(counter)
    pipeline=SMOTE(sampling_strategy=0.99, k_neighbors=21)

    Xtrain, ytrain = pipeline.fit_resample(Xtrain, ytrain)
    # summarize the new class distribution
    counter = Counter(ytrain)
    print(counter)


    
    return Xtrain,ytrain

#GaussianKernels used when testing different weighting options   
def gaussian_kernel10(distances):
    weights = np.exp(-10*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel100(distances):
    weights = np.exp(-100*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel1000(distances):
    weights = np.exp(-1000*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel10000(distances):
    weights = np.exp(-10000*(distances**2))
    return weights/np.sum(weights)









#######Functions End

#Main
def main():

    #KNN details
    # weighting="uniform"
    weighting="distance"
    # weighting=gaussian_kernel10
    k_value=5
        
    #Gathering Data
    df = pd.read_csv('dataset.csv').iloc[:, 1:]


    #Defining features and target classes
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)

    #Our Test and Training Data
    Xtrain, Xtest, ytrain, ytest = train_test_split(features,target,test_size=0.2)
    #Rebalance Training Data using Smote
    Xtrain,ytrain=dataRebalance(Xtrain,ytrain)

    #########################
    #Cross Validation for KNN
    Ki_range= [1, 5, 10, 15, 20, 25, 50, 100]
    chooseKValue(Ki_range,Xtrain,ytrain,weighting)

    Ki_range= [2,4,5,6,8,10]
    chooseKValue(Ki_range,Xtrain,ytrain,weighting)
    #########################

    ##############Model Results
    #Knn model defined here
    modelKnn = KNeighborsClassifier(n_neighbors=k_value,weights=weighting).fit(Xtrain, ytrain)

    #Predictions
    yPredictions = modelKnn.predict(Xtest)

    print("Knn Information Data Report")
    print(confusion_matrix(ytest, yPredictions))
    print(classification_report(ytest, yPredictions))

    #Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(ytest, yPredictions).ravel()
    print(tn, fp, fn, tp)

    #Printing the Accuracy Score
    print("accuracy_score : ",accuracy_score(ytest, yPredictions))
    print("f1_score : ", f1_score(ytest, yPredictions))

    #Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    ydummy = dummy.predict(Xtest)

    print("Dummy Data")
    print(confusion_matrix(ytest, ydummy))
    print(classification_report(ytest, ydummy))

    #Printing the Accuracy Score
    print("accuracy_score : ",accuracy_score(ytest, ydummy))
    print("f1_score : ",f1_score(ytest, ydummy))

    #Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(ytest, ydummy).ravel()
    print(tn, fp, fn, tp)


    #Knn Model
    ConfusionMatrixDisplay.from_predictions(ytest, yPredictions)
    plt.show()
    #Dummy Model
    ConfusionMatrixDisplay.from_predictions(ytest, ydummy)
    plt.show()


    #AUC results
    print("AUC Results")
    knnAUC = roc_auc_score(ytest, yPredictions)
    print("knnAUC: ",knnAUC)
    dummyAUC = roc_auc_score(ytest, ydummy)
    print("dummyAUC: ",dummyAUC)
    




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



