from pathlib import Path

from src.specs import DISEASE_SPECS

ROOT = Path(__file__).resolve().parents[1]


def test_required_dataset_files_exist() -> None:
    missing = [
        spec.dataset_filename
        for spec in DISEASE_SPECS.values()
        if not (ROOT / "data" / spec.dataset_filename).is_file()
    ]
    assert not missing, f"Missing dataset files in data/: {missing}"


def test_required_model_files_exist() -> None:
    missing = [
        spec.model_filename
        for spec in DISEASE_SPECS.values()
        if not (ROOT / "models" / spec.model_filename).is_file()
    ]
    assert not missing, f"Missing model files in models/: {missing}"


def test_key_documentation_files_exist() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "docs" / "DATA_DICTIONARY.md",
        ROOT / "docs" / "MODEL_CARD.md",
    ]
    assert all(path.is_file() for path in required)
