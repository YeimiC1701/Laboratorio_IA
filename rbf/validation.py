import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split


# Hold Out 70/30
def hold_out(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)

# 10-Fold Cross-Validation
def ten_fold_cross_validation(X, y, model):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        accuracies.append(model.score(X_test, y_test))
    return np.mean(accuracies)

# Leave-One-Out Cross-Validation
def leave_one_out(X, y, model):
    loo = LeaveOneOut()
    accuracies = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        accuracies.append(model.score(X_test, y_test))
    return np.mean(accuracies)
