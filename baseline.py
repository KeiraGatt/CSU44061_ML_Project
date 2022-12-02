# Keira Gatt (#19334557)
# CSU44061 Group Project
# 02.12.22

# Create a Baseline Model

from sklearn.dummy import DummyClassifier


# Dummy classifier
def baseline(x_train, y_train, x_val):
    # Dummy classifier that always predicts most frequent class in y_train
    dummy_clf = DummyClassifier(strategy='most_frequent')
    dummy_clf.fit(x_train, y_train)
    dummy_pred = dummy_clf.predict(x_val)
    #print(dummy_pred)

    return [dummy_pred, dummy_clf]

