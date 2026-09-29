import pickle

import pytest

from src.model_io import load_model


def test_load_model_reads_trusted_local_pickle(tmp_path):
    model = {"name": "test-model"}
    path = tmp_path / "model.sav"
    with path.open("wb") as file:
        pickle.dump(model, file)

    assert load_model("model.sav", model_dir=tmp_path) == model


def test_load_model_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model("missing.sav", model_dir=tmp_path)
