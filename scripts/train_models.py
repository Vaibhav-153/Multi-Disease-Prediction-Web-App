"""Reproduce the three baseline classifiers from the project datasets."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def evaluate(y_true, y_pred) -> dict[str, object]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_model(model, filename: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with (MODEL_DIR / filename).open("wb") as file:
        pickle.dump(model, file)


def train_diabetes() -> dict[str, object]:
    data = pd.read_csv(DATA_DIR / "diabetes.csv")
    x = data.drop(columns="Outcome")
    y = data["Outcome"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = svm.SVC(kernel="linear")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    save_model(model, "diabetes_model.sav")
    return evaluate(y_test, predictions)


def train_heart() -> dict[str, object]:
    data = pd.read_csv(DATA_DIR / "heart.csv", encoding="utf-8-sig")
    x = data.drop(columns="target")
    y = data["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2,
    )

    # The original notebook reached the default iteration limit. Increasing
    # max_iter fixes convergence without changing the model family.
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    save_model(model, "heart_disease_model.sav")
    return evaluate(y_test, predictions)


def train_parkinsons() -> dict[str, object]:
    data = pd.read_csv(DATA_DIR / "parkinsons.csv")
    x = data.drop(columns=["name", "status"])
    y = data["status"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=2,
    )

    model = svm.SVC(kernel="linear")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    save_model(model, "parkinsons_model.sav")
    return evaluate(y_test, predictions)


def main() -> None:
    metrics = {
        "diabetes": train_diabetes(),
        "heart_disease": train_heart(),
        "parkinsons": train_parkinsons(),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
