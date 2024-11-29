from sklearn import datasets
from sklearn.datasets import load_iris


# Cargar dataset Iris
def load_iris_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

# Cargar del dataset (Breast Cancer)
def load_breast_cancer_data():
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    return X, y

# Carga del dataset (Wine)
def load_wine_data():
    wine = datasets.load_wine()
    X = wine.data
    y = wine.target
    return X, y
