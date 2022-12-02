# Keira Gatt (#19334557)
# CSU44061 Group Project
# 02.12.22

# Analyse Feature Correlation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

# If p-value ≥ 0.05 ,failed to reject null hypothesis there is no any relationship between target variable and categorical feature
# If p_value < 0.05 ,Rejects null hypothesis and there will be some relationship between target variable and categorical feature

# Returns features NOT to include
def feature_selection():
    # Read in DF & drop index
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)

    # Get Chi-scores
    chi_scores = chi2(features, target)

    # Extract p-vals
    p_values = pd.Series(chi_scores[1], index=features.columns)
    p_val_list = p_values.tolist()
    feature_idx = []
    idx = 0
    # If p-val >= 0.05, add to list to remove from dataset
    for p_val in p_val_list:
        if p_val >= 0.05:
            feature_idx.append(idx)
        idx += 1

    p_values.plot.bar()
    plt.title('P-Value vs. Feature')
    plt.xlabel('Feature Name')
    plt.ylabel('P-Value')
    plt.tight_layout()
    plt.show()

    return feature_idx