# Keira Gatt (#19334557)
# CSU44061 Group Project
# 02.12.22

# Use Cross-Validation to do Feature Engineering (PolynomialFeatures)
# & find best weight for L1 Penalty

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from balance_data import *


def l1():
    # Read in DF & drop index
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)
    features = features.to_numpy()

    # Split data into training 80% & validation set 20%
    x_train, x_val, y_train, y_val = train_test_split(features, target, train_size=0.80, random_state=1)

    # Set max degree for poly features & range for hyper param C
    max_deg = 3
    c_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    # Don't use degree = 0
    for d in [x for x in range(max_deg) if x != 0]:
        mean_error = []
        std_error = []
        for c in c_range:
            # Compute the polynomial features for the current degree
            poly = PolynomialFeatures(d, include_bias=False)

            # Note this sets the 1st index of the set to 0
            x_poly = poly.fit_transform(x_train)

            # Reset the index of set to start at 0
            y_train = y_train.reset_index(drop = True)

            model = LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=10000)

            temp = []
            kf = KFold(n_splits=5)
            for train, test in kf.split(x_poly):

                # Only over-sample the test data
                oversampled_data = apply_SMOTE(x_poly[train], y_train[train])
                x_train_oversampled = oversampled_data[0]
                y_train_oversampled = oversampled_data[1]

                # Fit the model with the training section of the data
                model.fit(x_train_oversampled, y_train_oversampled)

                # Make predictions with the test part of data (not oversampled)
                target_pred = model.predict(x_poly[test])

                # Calculate F1 score & append to array for current C value
                temp.append(f1_score(y_train[test], target_pred))

            mean_error.append(np.array(temp).mean())
            print('F1-Score - %f' % (np.array(temp).mean()))
            std_error.append(np.array(temp).std())
            print('Model Co-eff for C = %f & Degree = %d' %(c, d))
            print(model.coef_)


        # Plot the vals for every val of d
        if d == 1:
            plt.errorbar(c_range, mean_error, yerr=std_error, color='r', label='Degree 1')
        elif d == 2:
            plt.errorbar(c_range, mean_error, yerr=std_error, color='b', label='Degree 2')

        plt.xlabel('C')
        plt.ylabel('F1 Score')
        plt.yticks(np.arange(0, 1, step=0.1))

    plt.title('L1 - Mean and Standard Deviation of the F1 Score vs C')
    plt.legend()
    #plt.savefig('L1- Cross-Val')
    plt.show()


def main():
    l1()


if __name__ == '__main__':
    main()
