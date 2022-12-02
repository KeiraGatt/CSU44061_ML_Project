# Keira Gatt (#19334557)
# CSU44061 Group Project
# 02.12.22

# Compare Chosen Model with Baseline using performance metrics specified in report

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from baseline import *
from balance_data import *


def compare_models():
    # Read in DF & drop index
    df = pd.read_csv('dataset.csv').iloc[:, 1:]
    target = df['chargeable']
    features = df.drop(['chargeable'], axis=1)

    x_train, x_val, y_train, y_val = train_test_split(features, target, train_size=0.80, random_state=1)
    x_train = x_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    x_train_np = x_train.to_numpy()
    x_val_np = x_val.to_numpy()

    # Only over-sample the test data
    oversampled_data = apply_SMOTE(x_train, y_train)
    x_train_oversampled = oversampled_data[0]
    y_train_oversampled = oversampled_data[1]

    # Fit with L1 Penalty Model
    model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=10000)
    model.fit(x_train_oversampled, y_train_oversampled)
    target_preds = model.predict(x_val_np)

    # Get Baseline Pred to compare performance
    base_res = baseline(x_train_oversampled, y_train_oversampled, x_val_np)
    base_preds = base_res[0]
    base_model = base_res[1]

    # Get conf matrices for all models
    get_conf_mat(y_val, target_preds, 'L1 Model')
    get_conf_mat(y_val, base_preds, 'Base Model')

    # Plot Confusion Matrices for Model and Baseline
    plot_conf_mats_bar(y_val, target_preds, base_preds, 'L1 Model Conf Mat Results', 'Baseline Model Conf Mat Results')

    # Plot ROC curves for models
    get_roc_curve(y_val, model, x_val, 'ROC for L1 Model')
    get_roc_curve(y_val, base_model, x_val, 'ROC for Base Model')

    # Get F1 core for all models
    get_F1_score(y_val, target_preds, 'L1 Model')
    get_F1_score(y_val, base_preds, 'Base Model')


# Create Confusion Matrix
def get_conf_mat(y_val, target_preds, title):
    ConfusionMatrixDisplay.from_predictions(y_val, target_preds)
    plt.title(title)
    plt.show()

# Plot confusion matrices as bar plots
def plot_conf_mats_bar(y_val, model_preds, base_preds, t1, t2):

    model_conf_vals = confusion_matrix(y_val, model_preds)
    m_tn, m_fp, m_fn, m_tp = model_conf_vals.ravel()

    base_conf_vals = confusion_matrix(y_val, base_preds)
    b_tn, b_fp, b_fn, b_tp = base_conf_vals.ravel()

    names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']

    fig, axs = plt.subplots(1, 2)
    axs[0].bar(names, [m_tn, m_fp, m_fn, m_tp])
    axs[0].set_title(t1)
    axs[0].set_ylabel('# of Predictions')
    axs[1].bar(names, [b_tn, b_fp, b_fn, b_tp])
    axs[1].set_title(t2)
    axs[1].set_ylabel('# of Predictions')
    plt.show()


# Calculate ROC curve
# True positive rate vs false positive rate
def get_roc_curve(y_val, model, x_val, title):

    y_scores = model.predict_proba(x_val)

    # Take all the rows of the second column with [:, 1] to only select the probability estimates of the positive class
    fpr, tpr, threshold = roc_curve(y_val, y_scores[:, 1])

    print(title + '- AUC = %f' % (auc(fpr, tpr)))

    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.title(title)
    #plt.savefig(title)
    plt.show()


# Print the F1-Score
def get_F1_score(y_val, target_pred, title):
    print('F1-Score for ' + title + ':%f' %(f1_score(y_val, target_pred)))


def main():
    compare_models()


if __name__ == '__main__':
    main()
