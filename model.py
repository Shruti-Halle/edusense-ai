"""Scikit-learn models for the EduSense demo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from data_utils import FEATURE_COLUMNS


def _preprocessor() -> ColumnTransformer:
    categorical = ["internet", "schoolsup", "famsup", "higher"]
    numeric = [column for column in FEATURE_COLUMNS if column not in categorical]
    return ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])


def train_models(data: pd.DataFrame) -> dict:
    """Train balanced logistic and random forest models and return diagnostics."""

    clean = data.copy()
    X = clean[FEATURE_COLUMNS].copy()
    y = clean["at_risk"].astype(int)
    if y.nunique() < 2:
        raise ValueError("The target needs both at-risk and not-at-risk rows.")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    models = {
        "Logistic regression": Pipeline([
            ("prep", _preprocessor()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "Random forest": Pipeline([
            ("prep", _preprocessor()),
            ("model", RandomForestClassifier(n_estimators=180, max_depth=6, class_weight="balanced", random_state=42)),
        ]),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "precision": precision_score(y_test, predictions, zero_division=0),
        }
    best_model_name = max(metrics, key=lambda name: metrics[name]["f1"])
    best_model = models[best_model_name]
    importances = feature_importance(best_model, X.columns)
    return {
        "models": models,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "metrics": metrics,
        "feature_importances": importances,
    }


def feature_importance(model: Pipeline, feature_names: pd.Index) -> dict[str, float]:
    """Aggregate one-hot encoded importances back to the original feature names."""

    estimator = model.named_steps["model"]
    encoded_names = model.named_steps["prep"].get_feature_names_out()
    raw = np.abs(estimator.coef_[0]) if hasattr(estimator, "coef_") else np.abs(estimator.feature_importances_)
    output: dict[str, float] = {name: 0.0 for name in feature_names}
    for name, value in zip(encoded_names, raw):
        clean_name = name.split("__", 1)[-1]
        matching = next((feature for feature in feature_names if clean_name.startswith(feature)), clean_name)
        output[matching] = output.get(matching, 0.0) + float(value)
    total = sum(output.values()) or 1.0
    return {key: value / total for key, value in output.items()}