import pandas as pd
import pytest

from scripts.train_models import validate_dataset
from src.specs import DISEASE_SPECS, feature_columns


def make_valid_frame(spec_name: str) -> pd.DataFrame:
    spec = DISEASE_SPECS[spec_name]
    columns = {column: [0.0, 1.0] for column in feature_columns(spec)}
    columns[spec.target_column] = [0, 1]
    return pd.DataFrame(columns)


def test_validate_dataset_accepts_required_schema() -> None:
    spec = DISEASE_SPECS["Diabetes"]
    validate_dataset(make_valid_frame("Diabetes"), spec)


def test_validate_dataset_rejects_missing_column() -> None:
    spec = DISEASE_SPECS["Heart Disease"]
    data = make_valid_frame("Heart Disease").drop(columns=[feature_columns(spec)[0]])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(data, spec)


def test_validate_dataset_rejects_single_class_target() -> None:
    spec = DISEASE_SPECS["Parkinson's"]
    data = make_valid_frame("Parkinson's")
    data[spec.target_column] = 1

    with pytest.raises(ValueError, match="must contain two classes"):
        validate_dataset(data, spec)
