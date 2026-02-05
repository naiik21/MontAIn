import pandas as pd

# =====================
# 1. Cargar dataset
# =====================
df = pd.read_csv("dataset.csv")

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
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =====================
# 4. RandomForest baseline
# =====================
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =====================
# 5. Evaluación
# =====================
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

print("Balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# =====================
# 6. Feature importance
# =====================
import numpy as np

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature importance:")
for i in indices:
    print(f"{feature_cols[i]}: {importances[i]:.3f}")
