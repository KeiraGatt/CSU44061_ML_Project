#Aaron Readman
#Dummy classifier model(Most Frequent)

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


def main():
        
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)

    #Our Test and Training Data
    Xtrain, Xtest, ytrain, ytest = train_test_split(features,target,test_size=0.2)

    #Dummy Classifier
    dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
    ydummy = dummy.predict(Xtest)

    print("Dummy Data")
    print(confusion_matrix(ytest, ydummy))
    print(classification_report(ytest, ydummy))

    #Printing the Accuracy Score
    print(accuracy_score(ytest, ydummy))

    #Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(ytest, ydummy).ravel()
    print(tn, fp, fn, tp)

if __name__ == '__main__':
    main()

