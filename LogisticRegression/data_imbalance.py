# Keira Gatt (#19334557)
# CSU44061 Group Project
# 02.12.22

# Analyse data imbalance & correct using SMOTE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import  f1_score
from sklearn.linear_model import LogisticRegression
import numpy as np

# Check how many chargeable and non-chargeable points
def check_data_balance():

    # Read in DF & drop index
    df = pd.read_csv('dataset.csv').iloc[:, 1:]

    # Split df into chargeable and non-chargeable tickets
    df_charge = df[df['chargeable'] == 1]
    df_no_charge = df[df['chargeable'] == -1]

    fig = plt.figure()
    labels = ['Chargeable', 'Non-Chargeable']
    ticket_num = [len(df_charge), len(df_no_charge)]
    plt.bar(labels, ticket_num)
    plt.ylabel('Number of Tickets')
    plt.title('Chargeable vs. Non-Chargeable Tickets')
    plt.savefig('Data Imbalance')
    plt.show()

# Use Cross-Validation to select best K value for SMOTE
def select_SMOTE_k():
    # Read in DF & drop index
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)
    features = features.to_numpy()

    # Range of K values to try
    k_range = [3, 5, 7, 9, 13, 15, 17, 19, 21, 31, 51, 91]

    mean_error = []
    std_error = []
    for k in k_range:
        kf = KFold(n_splits=5)
        model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=10000)

        temp = []
        for train, test in kf.split(features):
            # Use SMOTE to synthesize samples in training set
            sm = SMOTE(sampling_strategy='auto', k_neighbors=k)
            x_oversampled, y_oversampled = sm.fit_resample(features[train], target[train])

            # Fit the model with the training section of the data (over-sampled)
            model.fit(x_oversampled, y_oversampled)
            # Make predictions with the test part of data (this part of data has not been over-sampled)
            target_pred = model.predict(features[test])
            # Calculate F1 score & append to array
            temp.append(f1_score(target[test], target_pred))

        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    # Plot the F1-Score for every val of K
    plt.errorbar(k_range, mean_error, yerr=std_error)
    plt.yticks(np.arange(0.7, 1, step=0.2))
    plt.xlabel('K Values')
    plt.ylabel('F1 Score')

    plt.title('Mean and Standard Deviation of the F1 Score vs SMOTE K Values')
    plt.show()

def main():
    check_data_balance()
    select_SMOTE_k()


if __name__ == '__main__':
    main()
