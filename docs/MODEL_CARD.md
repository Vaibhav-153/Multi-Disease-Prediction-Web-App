# Model Card

## Purpose

This repository contains three small binary classifiers used in an educational Streamlit application:

- diabetes classification
- heart-disease classification
- Parkinson's disease classification

The application is a learning and portfolio project. It is not a medical device and must not be used for diagnosis, treatment, triage, screening decisions, or other clinical decisions.

## Models

| Task | Baseline model | Features | Target |
| --- | --- | ---: | --- |
| Diabetes | `SVC(kernel="linear")` | 8 | `Outcome` |
| Heart disease | `LogisticRegression(max_iter=1000)` | 13 | `target` |
| Parkinson's | `SVC(kernel="linear")` | 22 | `status` |

The refactored training code keeps the model families used by the original project. The heart-disease model uses a higher `max_iter` because the historical notebook reached the default logistic-regression iteration limit.

## Input definitions

Input names, units, and categorical encodings are documented in `docs/DATA_DICTIONARY.md` and are also shown inside the Streamlit interface.

The feature order is defined in `src/specs.py`. Training and inference use that same metadata to reduce the chance of feature-order mismatches.

## Training workflow

The maintained training entry point is:

```bash
python -m scripts.train_models
```

For each dataset, the script:

1. checks that required feature and target columns exist;
2. selects model features in a fixed order;
3. performs the project's train/test split;
4. fits the baseline estimator;
5. evaluates the held-out rows;
6. saves the model to `models/`;
7. writes reproducibility metadata and metrics to `reports/metrics.json`.

The metrics report includes:

- accuracy
- balanced accuracy
- precision
- recall
- F1 score
- ROC-AUC
- confusion matrix
- train/test row counts
- class counts
- dataset SHA-256 digest
- Python, pandas, and scikit-learn versions

## Historical notebook results

The original notebooks reported approximately:

| Task | Test accuracy |
| --- | ---: |
| Diabetes | 0.7532 |
| Heart disease | 0.8197 |
| Parkinson's | 0.8718 |

These values come from single train/test splits. They are baseline development results, not clinical performance estimates.

## Evaluation limitations

The project still has important limitations:

- small public datasets;
- no external clinical validation;
- no confidence intervals;
- no probability calibration;
- no clinically selected decision thresholds;
- no subgroup/fairness analysis;
- no prospective evaluation;
- no comparison with a clinically meaningful baseline.

The Parkinson's dataset contains multiple recordings from subjects. A random row-level split can therefore place recordings associated with the same person in both training and test sets. A stronger experiment should use subject-level grouped splitting.

The diabetes dataset contains zero values in measurement fields where zero may indicate missing or invalid observations. The baseline pipeline preserves the historical preprocessing rather than changing those values without a documented experiment.

The two SVM baselines are not feature-scaled. A future experiment should compare preprocessing pipelines using cross-validation rather than modifying the existing baseline silently.

## Intended use

Appropriate uses:

- learning how to structure a small ML repository;
- demonstrating tabular classification and model deployment;
- comparing baseline evaluation approaches;
- discussing model limitations in interviews.

Out-of-scope uses:

- diagnosing disease;
- medical screening or triage;
- treatment decisions;
- interpreting an individual's health status;
- production healthcare deployment.

## Artifact and security notes

Model files use Python pickle (`.sav`). Pickle can execute code during loading, so model artifacts must only come from trusted sources.

Pickled scikit-learn models are also sensitive to Python and library versions. If an artifact fails to load, recreate it with the documented training environment instead of downloading an unknown replacement.
