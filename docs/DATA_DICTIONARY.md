# Data Dictionary

This project uses three small tabular datasets. The application expects values in the same order and encoding used by the training CSV files.

The tables below document the fields used by the models. They are included so that the Streamlit form, training script, and README all use the same definitions.

## Diabetes dataset

The repository contains 768 rows with eight model inputs and the binary target `Outcome`. The schema matches the commonly distributed Pima Indians Diabetes dataset. The exact original download URL used for this repository is not recorded, so the project should not claim a specific copy as its provenance until that is confirmed.

| Column | Meaning | Type / unit |
| --- | --- | --- |
| `Pregnancies` | Number of times pregnant | integer count |
| `Glucose` | Plasma glucose concentration recorded by the source dataset | numeric |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skin-fold thickness | mm |
| `Insulin` | 2-hour serum insulin | mu U/ml |
| `BMI` | Body mass index | kg/m^2 |
| `DiabetesPedigreeFunction` | Dataset-specific family-history score | numeric |
| `Age` | Age | years |
| `Outcome` | Binary target | 0 / 1 |

### Data-quality note

The CSV contains zero values in fields where zero may represent a missing or invalid measurement, including glucose, blood pressure, skin thickness, insulin, and BMI. The baseline training script preserves the historical project behavior and does not silently impute these values.

## Heart-disease dataset

The repository contains a 303-row Cleveland-style heart-disease table with 13 model inputs and a binary `target`.

The original UCI Heart Disease documentation uses four chest-pain categories and defines the medical meaning of the fields. The local CSV uses common recoded integer values for several categorical columns.

| Column | Meaning | Local encoding / unit |
| --- | --- | --- |
| `age` | Age | years |
| `sex` | Sex code | 0 = female, 1 = male |
| `cp` | Chest-pain type | 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic |
| `trestbps` | Resting blood pressure | mm Hg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0 = false, 1 = true |
| `restecg` | Resting ECG result | 0 = normal, 1 = ST-T abnormality, 2 = probable/definite LV hypertrophy |
| `thalach` | Maximum heart rate achieved | bpm |
| `exang` | Exercise-induced angina | 0 = no, 1 = yes |
| `oldpeak` | ST depression induced by exercise relative to rest | numeric |
| `slope` | Peak exercise ST-segment slope | 0 = upsloping, 1 = flat, 2 = downsloping |
| `ca` | Major vessels colored by fluoroscopy | local CSV contains 0-4; original UCI documentation defines 0-3 |
| `thal` | Recoded thal field | 0 = unknown/not recorded, 1 = fixed defect, 2 = normal, 3 = reversible defect |
| `target` | Binary target used by this repository | 0 / 1 |

### Encoding note

The original UCI coding is not identical to every cleaned CSV distributed online. The application deliberately follows the values in this repository's `data/heart.csv` so that inference matches model training.

## Parkinson's dataset

The Parkinson's table contains biomedical voice measurements. `name` identifies a recording/subject and is excluded from model inputs. `status` is the binary target.

| Column | Meaning |
| --- | --- |
| `name` | Subject/recording identifier; not used as a model feature |
| `MDVP:Fo(Hz)` | Average vocal fundamental frequency |
| `MDVP:Fhi(Hz)` | Maximum vocal fundamental frequency |
| `MDVP:Flo(Hz)` | Minimum vocal fundamental frequency |
| `MDVP:Jitter(%)` | Variation in fundamental frequency, percentage measure |
| `MDVP:Jitter(Abs)` | Absolute variation in fundamental frequency |
| `MDVP:RAP` | Relative average perturbation measure of frequency variation |
| `MDVP:PPQ` | Pitch perturbation quotient measure of frequency variation |
| `Jitter:DDP` | Difference-of-differences period jitter measure |
| `MDVP:Shimmer` | Variation in amplitude |
| `MDVP:Shimmer(dB)` | Variation in amplitude in decibels |
| `Shimmer:APQ3` | Three-point amplitude perturbation quotient |
| `Shimmer:APQ5` | Five-point amplitude perturbation quotient |
| `MDVP:APQ` | MDVP amplitude perturbation quotient |
| `Shimmer:DDA` | Difference-of-differences amplitude measure |
| `NHR` | Noise-to-harmonics ratio |
| `HNR` | Harmonics-to-noise ratio |
| `RPDE` | Nonlinear dynamical complexity measure |
| `DFA` | Signal fractal scaling exponent |
| `spread1` | Nonlinear measure of fundamental-frequency variation |
| `spread2` | Nonlinear measure of fundamental-frequency variation |
| `D2` | Nonlinear dynamical complexity measure |
| `PPE` | Nonlinear measure of fundamental-frequency variation |
| `status` | Binary target: 1 = Parkinson's, 0 = healthy in the source dataset |

## Source references

- UCI Machine Learning Repository: Heart Disease dataset
- UCI Machine Learning Repository: Parkinson's dataset

The exact source used for the repository's diabetes CSV should be added once it can be confirmed from the original download history or project notes.
