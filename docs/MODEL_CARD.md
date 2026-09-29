# Model Card

## Purpose

This repository contains three small binary-classification models used in an educational Streamlit application:

- diabetes classification
- heart-disease classification
- Parkinson's classification

The application is a machine-learning demonstration. It is not a medical device and must not be used for diagnosis, treatment, triage, or other clinical decisions.

## Models

| Task | Baseline model | Features | Target |
| --- | --- | ---: | --- |
| Diabetes | Linear SVM | 8 | `Outcome` |
| Heart disease | Logistic regression | 13 | `target` |
| Parkinson's | Linear SVM | 22 | `status` |

The refactored training script keeps the original model families. The heart-disease logistic regression uses a higher `max_iter` value because the original notebook reached the default iteration limit.

## Recorded notebook results

The historical notebooks in the original repository report approximately:

- Diabetes test accuracy: 0.7532
- Heart-disease test accuracy: 0.8197
- Parkinson's test accuracy: 0.8718

These values come from single train/test splits and should not be treated as clinical performance estimates.

## Evaluation limitations

The original notebooks primarily report accuracy. A medical classification project should also consider precision, recall, F1 score, confusion matrices, calibration, subgroup performance, external validation, and clinically appropriate decision thresholds.

The new training script records accuracy, precision, recall, F1 score, and a confusion matrix for each task, but it still uses the project's small public datasets and single holdout splits.

## Known limitations

- Small datasets.
- No external clinical validation.
- No probability calibration or confidence intervals.
- No subgroup/fairness analysis.
- No missing-value strategy beyond what is already present in the datasets.
- The diabetes dataset contains physiologically implausible zero values in some measurement fields; the baseline script preserves the historical project behavior rather than silently changing preprocessing.
- The Parkinson's dataset contains multiple recordings from subjects, so random row-level splitting can place recordings from the same person in both train and test sets. A stronger evaluation would split by subject.
- Pickled model artifacts are Python-version and library-version sensitive and must only be loaded from trusted sources.
