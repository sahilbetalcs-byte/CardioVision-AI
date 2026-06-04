import json
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


RANDOM_STATE = 42


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, object]:
    return {
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


df = pd.read_csv("Heart_Disease_Dataset.csv")
column_names = [
    "age",
    "gender",
    "chestpain",
    "restingBP",
    "serumcholestrol",
    "fastingbloodsugar",
    "restingrelectro",
    "maxheartrate",
    "exerciseangia",
    "oldpeak",
    "slope",
    "noofmajorvessels",
    "target",
]
df.columns = column_names

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, max_depth=8
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    ),
    "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
    "Neural Network": MLPClassifier(
        hidden_layer_sizes=(32, 16), activation="relu", max_iter=2000, random_state=RANDOM_STATE
    ),
}

os.makedirs("models", exist_ok=True)

metrics = {}
registry = {}

for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics[model_name] = compute_metrics(y_test, y_pred, y_proba)

    model_filename = model_name.lower().replace(" ", "_") + ".pkl"
    model_path = os.path.join("models", model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    registry[model_name] = model_filename

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

best_model_name = max(
    metrics.items(), key=lambda item: (item[1]["f1"], item[1]["roc_auc"])
)[0]

metadata = {
    "best_model": best_model_name,
    "models": registry,
    "metrics": metrics,
    "features": X.columns.tolist(),
    "dataset_size": int(len(df)),
    "class_balance": {
        "no_heart_disease": int((y == 0).sum()),
        "heart_disease": int((y == 1).sum()),
    },
}

with open("models/model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Training complete. Best model: {best_model_name}")