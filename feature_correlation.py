# Keira Gatt
# 23.11.22
# Analyse Feature Correlation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2

# If p-value ≥ 0.05 ,failed to reject null hypothesis there is no any relationship between target variable and categorical feature
# if p_value < 0.05 ,Rejects null hypothesis and there will be some relationship between target variable and categorical feature
def chi_square():
    # Read in DF & drop index
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)

    chi_scores = chi2(features, target)
    p_values = pd.Series(chi_scores[1], index=features.columns)
    p_values.sort_values(ascending=False, inplace=True)
    p_values.plot.bar()
    plt.tight_layout()
    plt.show()

def main():
    chi_square()


if __name__ == '__main__':
    main()