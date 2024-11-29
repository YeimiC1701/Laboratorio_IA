from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_performance(model, X_test, y_test):
    """
    Calcula el Accuracy y la matriz de confusión.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm