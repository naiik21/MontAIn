import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def baseline(dataset_path="dataset.csv", test_size=0.25, random_state=42, show_plots=True):
    """
    Entrena y evalúa un modelo RandomForest baseline para clasificación de dificultad.
    
    Args:
        dataset_path (str): Ruta al archivo CSV del dataset
        test_size (float): Proporción del dataset para test (default: 0.25)
        random_state (int): Semilla para reproducibilidad (default: 42)
        show_plots (bool): Si mostrar gráficos (default: True)
    
    Returns:
        dict: Diccionario con el modelo entrenado, métricas y datos de evaluación
    """
    # =====================
    # 1. Cargar dataset
    # =====================
    df = pd.read_csv(dataset_path)

    # Mapear dificultad a entero
    difficulty_map = {
        "sendero fácil": 0,
        "moderado": 1,
        "difícil": 2,
        "alta montaña": 3,
        "alpinismo ligero": 4,
        "alpinismo técnico": 5
    }

    df["difficulty"] = df["difficulty"].map(difficulty_map)

    # Eliminar columnas no útiles
    df = df.drop(columns=["filename"])

    # Check
    assert df["difficulty"].isnull().sum() == 0

    # =====================
    # 2. Features / Target
    # =====================
    feature_cols = [
        "distance_km",
        "elevation_gain",
        "elevation_loss",
        "max_elevation",
        "min_elevation",
        "max_slope",
        "mean_slope",
        "pct_over_30",
        "pct_over_40",
        "pct_over_45",
        "mean_aspect",
        "rugosity_mean",
        "exposed_pct"
    ]

    X = df[feature_cols]
    y = df["difficulty"]

    assert X.isnull().sum().sum() == 0

    # =====================
    # 3. Train / Test split
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # =====================
    # 4. RandomForest baseline
    # =====================
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # =====================
    # 5. Evaluación
    # =====================
    y_pred = model.predict(X_test)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced accuracy:", balanced_acc)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    if show_plots:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

    # =====================
    # 6. Feature importance
    # =====================
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature importance:")
    feature_importance_dict = {}
    for i in indices:
        importance_value = importances[i]
        feature_name = feature_cols[i]
        print(f"{feature_name}: {importance_value:.3f}")
        feature_importance_dict[feature_name] = importance_value

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "feature_importance": feature_importance_dict,
        "feature_cols": feature_cols
    }


