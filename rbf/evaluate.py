from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, conf_matrix
