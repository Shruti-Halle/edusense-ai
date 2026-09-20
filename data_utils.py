"""Data loading and synthetic data helpers for EduSense AI."""

from __future__ import annotations

import csv
import io

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "studytime", "failures", "absences", "G1", "G2", "health", "freetime",
    "goout", "Medu", "Fedu", "internet", "schoolsup", "famsup", "higher", "age",
]

FEATURE_LABELS = {
    "studytime": "Study time",
    "failures": "Past failures",
    "absences": "Absences",
    "G1": "First period grade",
    "G2": "Second period grade",
    "health": "Health",
    "freetime": "Free time",
    "goout": "Going out",
    "Medu": "Mother's education",
    "Fedu": "Father's education",
    "internet": "Internet access",
    "schoolsup": "School support",
    "famsup": "Family support",
    "higher": "Higher-ed plans",
    "age": "Age",
}


def generate_synthetic_data(n_rows: int = 520, seed: int = 42) -> pd.DataFrame:
    """Create a realistic, reproducible student-performance dataset."""

    rng = np.random.default_rng(seed)
    studytime = rng.choice([1, 2, 3, 4], size=n_rows, p=[.18, .48, .27, .07])
    failures = np.clip(rng.poisson(.38, n_rows), 0, 3)
    absences = np.clip(rng.gamma(2.0, 4.8, n_rows).round().astype(int), 0, 32)
    health = rng.integers(1, 6, n_rows)
    freetime = rng.integers(1, 6, n_rows)
    goout = rng.integers(1, 6, n_rows)
    medu = rng.integers(0, 5, n_rows)
    fedu = rng.integers(0, 5, n_rows)
    age = rng.integers(15, 20, n_rows)
    internet = rng.choice(["yes", "no"], n_rows, p=[.83, .17])
    schoolsup = rng.choice(["yes", "no"], n_rows, p=[.13, .87])
    famsup = rng.choice(["yes", "no"], n_rows, p=[.62, .38])
    higher = rng.choice(["yes", "no"], n_rows, p=[.90, .10])
    sex = rng.choice(["F", "M"], n_rows)

    base = (
        11.4
        + studytime * 0.72
        + medu * 0.22
        + fedu * 0.14
        + (internet == "yes") * 0.35
        + (famsup == "yes") * 0.22
        + (higher == "yes") * 0.22
        - failures * 1.35
        - absences * 0.075
        + rng.normal(0, 1.55, n_rows)
    )
    g1 = np.clip(np.round(base + rng.normal(-.55, 1.65, n_rows)), 0, 20).astype(int)
    g2 = np.clip(np.round(base + rng.normal(.15, 1.4, n_rows)), 0, 20).astype(int)
    g3 = np.clip(np.round(g1 * .34 + g2 * .55 + rng.normal(0, 1.35, n_rows)), 0, 20).astype(int)

    frame = pd.DataFrame({
        "studytime": studytime, "failures": failures, "absences": absences,
        "G1": g1, "G2": g2, "G3": g3, "health": health, "freetime": freetime,
        "goout": goout, "Medu": medu, "Fedu": fedu, "internet": internet,
        "schoolsup": schoolsup, "famsup": famsup, "higher": higher, "age": age, "sex": sex,
    })
    frame["at_risk"] = (frame["G3"] < 10).astype(int)
    return frame


def detect_separator(sample: str) -> str:
    """Detect a CSV delimiter, defaulting to a comma when uncertain."""

    try:
        return csv.Sniffer().sniff(sample[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        return ";" if ";" in sample[:4096] else ","


def load_student_data(file_like: io.BytesIO, filename: str) -> pd.DataFrame:
    """Load an uploaded student file and normalize the target column safely."""

    raw = file_like.getvalue()
    sample = raw[:8192].decode("utf-8-sig", errors="ignore")
    separator = detect_separator(sample)
    frame = pd.read_csv(io.BytesIO(raw), sep=separator)
    frame.columns = [str(column).strip().strip('"') for column in frame.columns]
    frame = frame.replace({"yes": "yes", "no": "no", "nan": np.nan})
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].astype(str).str.strip().str.strip('"')
    numeric_columns = ["studytime", "failures", "absences", "G1", "G2", "G3", "health", "freetime", "goout", "Medu", "Fedu", "age"]
    for column in numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = frame[column].fillna(frame[column].median())
    for column in ["internet", "schoolsup", "famsup", "higher", "sex"]:
        if column in frame:
            frame[column] = frame[column].fillna(frame[column].mode().iloc[0] if not frame[column].mode().empty else "no")
    if "G3" not in frame:
        raise ValueError("The file needs a G3 final-grade column.")
    frame["at_risk"] = (frame["G3"] < 10).astype(int)
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0 if column not in {"internet", "schoolsup", "famsup", "higher"} else "no"
    return frame