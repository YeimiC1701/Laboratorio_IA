import pandas as pd
from RBF.metrics import evaluate_performance
from mlp_model import create_mlp
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from RBF.validation import (cross_validation, hold_out_validation,
                                leave_one_out_validation)


def run_experiment(dataset_name, X, y):
    """
    Realiza las validaciones Hold-Out, 10-Fold Cross-Validation y Leave-One-Out 
    sobre el dataset dado y muestra los resultados.
    """
    print(f"\n--- Dataset: {dataset_name} ---")

    # Crear el modelo MLP
    mlp = create_mlp(hidden_layers=(10, 5), max_iter=500)

    # Validación Hold-Out
    model, X_test, y_test = hold_out_validation(mlp, X, y)
    acc, cm = evaluate_performance(model, X_test, y_test)
    print("\nHold-Out Validation:")
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cm}")

    # Validación 10-Fold Cross-Validation
    cv_score = cross_validation(mlp, X, y, folds=10)
    print("\n10-Fold Cross-Validation:")
    print(f"Mean Accuracy: {cv_score}")

    # Validación Leave-One-Out
    loo_score = leave_one_out_validation(mlp, X, y)
    print("\nLeave-One-Out Validation:")
    print(f"Mean Accuracy: {loo_score}")

# Cargar los datasets
datasets = {
    "Iris": load_iris(),
    "Breast Cancer": load_breast_cancer(),
    "Wine": load_wine()
}

# Ejecutar experimentos para cada dataset
for name, dataset in datasets.items():
    X, y = dataset.data, dataset.target
    run_experiment(name, X, y)
