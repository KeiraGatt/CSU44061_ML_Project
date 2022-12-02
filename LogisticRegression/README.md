Keira Gatt (#19334557)
02.12.22

Use Logisitc Regression Classifier to predict if a ticket is chargeable/non-chargeable.

Run the following files to analyse the Logistic Regression Classifier -

L1.py - Use Cross-Validation to do Feature Engineering (PolynomialFeatures) & find best weight for L1 Penalty.
	Note SMOTE is applied to training dataset.

L2.py - Use Cross-Validation to do Feature Engineering (PolynomialFeatures) & find best weight for L2 Penalty.
	Note SMOTE is applied to training dataset & feature selection is performed via chi-square test.

compare_models.py - Compare the performance of the chosen L1 model with a Dummy Classifier.