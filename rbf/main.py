from datasets import load_breast_cancer_data, load_iris_data, load_wine_data
from evaluate import evaluate
from rbf_classifier import RBFClassifier
from validation import hold_out, leave_one_out, ten_fold_cross_validation

# Crear el clasificador RBF
model = RBFClassifier(gamma=0.5)

# Función para realizar validación y evaluación en cada dataset
def process_dataset(X, y, model, dataset_name):
    print(f"\nProcesando el dataset {dataset_name}:")

    # Hold Out 70/30
    X_train, X_test, y_train, y_test = hold_out(X, y)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy, conf_matrix = evaluate(y_test, y_pred)
    print(f"Hold Out 70/30:")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # 10-Fold Cross-Validation
    accuracy_10_fold = ten_fold_cross_validation(X, y, model)
    print(f"10-Fold Cross-Validation Accuracy: {accuracy_10_fold}")

    # Leave-One-Out Cross-Validation
    accuracy_loo = leave_one_out(X, y, model)
    print(f"Leave-One-Out Cross-Validation Accuracy: {accuracy_loo}")

# Cargar y procesar los tres datasets
datasets = [
    ("Iris", load_iris_data),
    ("Breast Cancer", load_breast_cancer_data),
    ("Wine", load_wine_data)
]

for dataset_name, load_function in datasets:
    X, y = load_function()  # Cargar el dataset
    process_dataset(X, y, model, dataset_name)
