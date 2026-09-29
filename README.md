# Multi-Disease Prediction Web App

A Streamlit application that runs three binary machine-learning classifiers for diabetes, heart disease, and Parkinson's disease.

This project is for learning and portfolio demonstration. The predictions are model outputs, not medical diagnoses, and should not be used for clinical decisions.

## What the project does

The app accepts numeric input features and sends them to one of three trained scikit-learn models:

| Task | Model | Input features |
| --- | --- | ---: |
| Diabetes | Linear SVM | 8 |
| Heart disease | Logistic regression | 13 |
| Parkinson's disease | Linear SVM | 22 |

The original notebooks reported test accuracies of approximately 75.3% for diabetes, 82.0% for heart disease, and 87.2% for Parkinson's disease. These results are from single holdout splits and are not clinical validation results.

## Project structure

```text
Multi-Disease-Prediction-Web-App/
├── .github/
│   └── workflows/
│       └── ci.yml
├── assets/
│   └── app-screenshot.png
├── data/
│   ├── diabetes.csv
│   ├── heart.csv
│   └── parkinsons.csv
├── docs/
│   ├── MODEL_CARD.md
│   └── project-report.pdf
├── models/
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   └── parkinsons_model.sav
├── notebooks/
│   └── archive/
│       ├── diabetes.ipynb
│       ├── heart_disease.ipynb
│       └── parkinsons.ipynb
├── reports/
│   └── .gitkeep
├── scripts/
│   └── train_models.py
├── src/
│   ├── __init__.py
│   ├── model_io.py
│   ├── prediction.py
│   └── specs.py
├── tests/
│   ├── test_prediction.py
│   └── test_specs.py
├── .gitignore
├── app.py
├── pyproject.toml
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## Datasets

### Diabetes

The repository contains 768 rows with eight numeric input features and a binary `Outcome` target. The schema matches the commonly used Pima Indians Diabetes dataset. The exact original download source should be recorded if available.

### Heart disease

The repository contains 303 rows, 13 input features, and a binary `target`. Its feature names match the commonly used processed Cleveland heart-disease schema (`age`, `sex`, `cp`, `trestbps`, `chol`, and related fields).

### Parkinson's disease

The repository contains biomedical voice measurements with 22 model features. `status` is the binary target and `name` identifies the recording/subject. The data corresponds to the Oxford Parkinson's Disease Detection dataset distributed through the UCI Machine Learning Repository.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

The application loads model artifacts from the local `models/` directory. No machine-specific absolute paths are required.

## Retrain the models

```bash
python -m scripts.train_models
```

The script reads the three CSV files from `data/`, trains the same baseline model families used in the original project, saves model artifacts into `models/`, and writes evaluation metrics to:

```text
reports/metrics.json
```

The generated report includes accuracy, precision, recall, F1 score, and a confusion matrix.

## Baseline method

### Diabetes

- Target: `Outcome`
- Split: 80/20
- Random state: 42
- Model: linear SVM

### Heart disease

- Target: `target`
- Split: 80/20 with stratification
- Random state: 2
- Model: logistic regression
- `max_iter=1000` is used to avoid the convergence warning produced by the original notebook.

### Parkinson's disease

- ID field excluded: `name`
- Target: `status`
- Split: 80/20
- Random state: 2
- Model: linear SVM

## Testing and linting

Install development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

Run tests:

```bash
python -m pytest -q
```

Run Ruff:

```bash
ruff check .
```

GitHub Actions runs both checks on pushes and pull requests.

## Important limitations

This project should not be presented as a diagnostic system.

- The datasets are small.
- Evaluation is based on simple train/test splits.
- The historical notebooks focus mainly on accuracy.
- No external clinical dataset is used for validation.
- No probability calibration or clinical threshold analysis is included.
- The Parkinson's dataset has repeated recordings from subjects; subject-level splitting would provide a stronger evaluation than a random row split.
- The baseline SVM models are not feature-scaled because the refactor preserves the original modeling approach. A future experiment should compare scaled pipelines using cross-validation.
- Some diabetes measurement fields contain zero values that may represent missing or invalid measurements; the baseline workflow does not correct them.

See `docs/MODEL_CARD.md` for additional notes.

## Future improvements

- move preprocessing and scaling into scikit-learn pipelines
- use stratified cross-validation where appropriate
- use subject-level grouping for Parkinson's evaluation
- compare precision, recall, F1, ROC-AUC, and calibration
- add probability outputs only after calibration and model review
- document exact dataset download sources and licences
- add model version metadata
- deploy the Streamlit app after validating the saved artifacts in a clean environment

## References

- UCI Machine Learning Repository — Heart Disease dataset
- UCI Machine Learning Repository — Parkinson's dataset

The exact source of the diabetes CSV should be added when confirmed from the original project notes or download history.
