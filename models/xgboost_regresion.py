from sklearn.metrics import mean_absolute_error
import numpy as np

# =====================
# 1. Modelo de regresión
# =====================
reg_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

reg_model.fit(X_train, y_train)

# =====================
# 2. Predicción continua
# =====================
y_pred_cont = reg_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_cont))

# =====================
# 3. Discretizar a clases
# =====================
bins = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
y_pred_class = np.digitize(y_pred_cont, bins) - 1

print("Balanced accuracy (ordinal):",
      balanced_accuracy_score(y_test, y_pred_class))

print(classification_report(y_test, y_pred_class))
