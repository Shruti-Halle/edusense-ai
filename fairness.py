"""Simple group-level fairness diagnostics for the EduSense demo."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from data_utils import FEATURE_COLUMNS


def _rows_for_attribute(data: pd.DataFrame, model, attribute: str) -> list[dict]:
    features = data[FEATURE_COLUMNS]
    predictions = model.predict(features)
    rows = []
    for value, group in data.groupby(attribute, dropna=False):
        indices = group.index
        truth = data.loc[indices, "at_risk"]
        predicted = predictions[data.index.get_indexer(indices)]
        rows.append({
            "attribute": attribute.title(),
            "group": str(value).title(),
            "students": len(group),
            "accuracy": accuracy_score(truth, predicted),
            "recall": recall_score(truth, predicted, zero_division=0),
        })
    return rows


def build_fairness_table(data: pd.DataFrame, model) -> pd.DataFrame:
    """Return accuracy and recall grouped by sex and internet access."""

    rows = _rows_for_attribute(data, model, "sex")
    rows.extend(_rows_for_attribute(data, model, "internet"))
    return pd.DataFrame(rows)