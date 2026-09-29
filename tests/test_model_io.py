import pickle
from pathlib import Path

import pytest

from src.model_io import load_model


def test_load_model_from_explicit_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "example.sav"
    with model_path.open("wb") as file:
        pickle.dump({"model": "test"}, file)

    assert load_model("example.sav", tmp_path) == {"model": "test"}


def test_load_model_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local .sav filename"):
        load_model("../example.sav", tmp_path)


def test_load_model_rejects_wrong_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local .sav filename"):
        load_model("model.pkl", tmp_path)


def test_load_model_reports_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model("missing.sav", tmp_path)
