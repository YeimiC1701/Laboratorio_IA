import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel


class RBFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.centers = X
        self.alpha = np.linalg.pinv(rbf_kernel(X, X, gamma=self.gamma)).dot(y)
        return self

    def predict(self, X):
        kernel_matrix = rbf_kernel(X, self.centers, gamma=self.gamma)
        return np.sign(kernel_matrix.dot(self.alpha))
