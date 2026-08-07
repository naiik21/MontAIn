import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    balanced_accuracy_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt


def xgboost(dataset_path="./dataset.csv", test_size=0.25, random_state=42, show_plots=True):
    """
    Entrena y evalúa un modelo XGBoost para clasificación de dificultad.
    
    Args:
        dataset_path (str): Ruta al archivo CSV del dataset
        test_size (float): Proporción del dataset para test (default: 0.25)
        random_state (int): Semilla para reproducibilidad (default: 42)
        show_plots (bool): Si mostrar gráficos (default: True)
    
    Returns:
        dict: Diccionario con el modelo entrenado, métricas y datos de evaluación
    """
    # =====================
    # 1. Dataset
    # =====================
    df = pd.read_csv(dataset_path)

    difficulty_map = {
        "sendero fácil": 0,
        "moderado": 1,
        "difícil": 2,
        "alta montaña": 3,
        "alpinismo ligero": 4,
        "alpinismo técnico": 5
    }

    df["difficulty"] = df["difficulty"].map(difficulty_map)
    df = df.drop(columns=["filename"])

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

    # =====================
    # 2. Split
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # =====================
    # 3. Modelo XGBoost
    # =====================
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=y.nunique(),
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state
    )

    model.fit(X_train, y_train)

    # =====================
    # 4. Evaluación
    # =====================
    y_pred = model.predict(X_test)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced accuracy:", balanced_acc)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    
    if show_plots:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - XGBoost")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # =====================
        # 5. Feature importance
        # =====================
        xgb.plot_importance(model, max_num_features=10)
        plt.show()

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
        "feature_cols": feature_cols
    }


