"""Reproduce the three baseline classifiers from the project datasets."""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
from pathlib import Path

import pandas as pd
import sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.specs import DISEASE_SPECS, DiseaseSpec, feature_columns

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(data: pd.DataFrame, spec: DiseaseSpec) -> None:
    """Validate required model columns before training."""
    required = {*feature_columns(spec), spec.target_column}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"{spec.dataset_filename} is missing required columns: {', '.join(missing)}"
        )

    if data.empty:
        raise ValueError(f"{spec.dataset_filename} contains no rows.")

    if data[spec.target_column].nunique(dropna=False) != 2:
        raise ValueError(
            f"{spec.dataset_filename} target '{spec.target_column}' must contain two classes."
        )


def evaluate(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    """Calculate classification metrics for one held-out split."""
    predictions = model.predict(x_test)
    decision_scores = model.decision_function(x_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def save_model(model, filename: str) -> Path:
    """Save a trained model into the repository model directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / filename
    with output_path.open("wb") as file:
        pickle.dump(model, file)
    return output_path


def load_training_data(spec: DiseaseSpec) -> tuple[pd.DataFrame, Path]:
    """Load and validate one repository dataset."""
    dataset_path = DATA_DIR / spec.dataset_filename
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    encoding = "utf-8-sig" if spec.key == "heart_disease" else "utf-8"
    data = pd.read_csv(dataset_path, encoding=encoding)
    validate_dataset(data, spec)
    return data, dataset_path


def build_report(
    *,
    spec: DiseaseSpec,
    data: pd.DataFrame,
    dataset_path: Path,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model,
    model_name: str,
    random_state: int,
) -> dict[str, object]:
    """Create reproducibility and evaluation metadata for one model."""
    return {
        "model": model_name,
        "features": list(feature_columns(spec)),
        "target": spec.target_column,
        "dataset_rows": int(len(data)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "random_state": random_state,
        "train_class_counts": {
            str(label): int(count) for label, count in y_train.value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(label): int(count) for label, count in y_test.value_counts().sort_index().items()
        },
        "dataset_sha256": file_sha256(dataset_path),
        "metrics": evaluate(model, x_test, y_test),
    }


def train_diabetes() -> dict[str, object]:
    """Train the diabetes baseline model."""
    spec = DISEASE_SPECS["Diabetes"]
    data, dataset_path = load_training_data(spec)
    x = data.loc[:, feature_columns(spec)]
    y = data[spec.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )
    model = svm.SVC(kernel="linear")
    model.fit(x_train, y_train)
    save_model(model, spec.model_filename)

    return build_report(
        spec=spec,
        data=data,
        dataset_path=dataset_path,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        model_name="SVC(kernel='linear')",
        random_state=42,
    )


def train_heart() -> dict[str, object]:
    """Train the heart-disease baseline model."""
    spec = DISEASE_SPECS["Heart Disease"]
    data, dataset_path = load_training_data(spec)
    x = data.loc[:, feature_columns(spec)]
    y = data[spec.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=2,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    save_model(model, spec.model_filename)

    return build_report(
        spec=spec,
        data=data,
        dataset_path=dataset_path,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        model_name="LogisticRegression(max_iter=1000)",
        random_state=2,
    )


def train_parkinsons() -> dict[str, object]:
    """Train the Parkinson's baseline model using the historical row-level split."""
    spec = DISEASE_SPECS["Parkinson's"]
    data, dataset_path = load_training_data(spec)
    x = data.loc[:, feature_columns(spec)]
    y = data[spec.target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=2,
    )
    model = svm.SVC(kernel="linear")
    model.fit(x_train, y_train)
    save_model(model, spec.model_filename)

    return build_report(
        spec=spec,
        data=data,
        dataset_path=dataset_path,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
        model_name="SVC(kernel='linear')",
        random_state=2,
    )


def main() -> None:
    """Train every baseline model and write a metrics report."""
    report = {
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "models": {
            "diabetes": train_diabetes(),
            "heart_disease": train_heart(),
            "parkinsons": train_parkinsons(),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "metrics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
