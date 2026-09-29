"""Model loading helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def load_model(filename: str, model_dir: Path | None = None) -> Any:
    """Load one trusted local pickle model artifact.

    Pickle files must only be loaded from trusted sources. This project loads
    artifacts committed with the repository or generated locally by the
    training script.
    """
    directory = model_dir or MODEL_DIR
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            "Ensure the saved model artifacts are present in the models directory."
        )

    with path.open("rb") as file:
        return pickle.load(file)
