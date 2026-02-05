import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score, classification_report


def xgboost_regresion(dataset_path="./dataset.csv", test_size=0.25, random_state=42):
    """
    Entrena y evalúa un modelo XGBoost de regresión para predecir dificultad,
    luego discretiza las predicciones a clases ordinales.
    
    Args:
        dataset_path (str): Ruta al archivo CSV del dataset
        test_size (float): Proporción del dataset para test (default: 0.25)
        random_state (int): Semilla para reproducibilidad (default: 42)
    
    Returns:
        dict: Diccionario con el modelo entrenado, métricas y datos de evaluación
    """
    # =====================
    # 1. Cargar y preparar datos
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # =====================
    # 2. Modelo de regresión
    # =====================
    reg_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )

    reg_model.fit(X_train, y_train)

    # =====================
    # 3. Predicción continua
    # =====================
    y_pred_cont = reg_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_cont)
    print("MAE:", mae)

    # =====================
    # 4. Discretizar a clases
    # =====================
    bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    y_pred_class = np.digitize(y_pred_cont, bins) - 1

    balanced_acc = balanced_accuracy_score(y_test, y_pred_class)
    print("Balanced accuracy (ordinal):", balanced_acc)
    print(classification_report(y_test, y_pred_class))


    reg_model.save_model('xgboost_regresion.model')
    
    return {
        "model": reg_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_continuous": y_pred_cont,
        "y_pred_class": y_pred_class,
        "mae": mae,
        "balanced_accuracy": balanced_acc,
        "feature_cols": feature_cols
    }



