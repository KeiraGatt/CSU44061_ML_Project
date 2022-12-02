# Keira Gatt (#19334557)
# CSU44061 Group Project
# 02.12.22

# Method to balance data using parameters selected by cross-validation
from imblearn.over_sampling import SMOTE


def apply_SMOTE(features, target):
    sm = SMOTE(sampling_strategy='auto', k_neighbors=21)
    x_oversampled, y_oversampled = sm.fit_resample(features, target)

    return [x_oversampled, y_oversampled]
