import numpy as np
from sklearn.model_selection import (LeaveOneOut, cross_val_score, train_test_split)


def hold_out_validation(model, X, y):
    """
    Realiza validación Hold-Out con división 70/30.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def cross_validation(model, X, y, folds=10):
    """
    Realiza validación con Cross-Validation k-fold.
    """
    scores = cross_val_score(model, X, y, cv=folds)
    return scores.mean()

def leave_one_out_validation(model, X, y):
    """
    Realiza validación Leave-One-Out.
    """
    loo = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=loo)
    return scores.mean()
