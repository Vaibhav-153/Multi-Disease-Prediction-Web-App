"""Metadata for model inputs and the Streamlit interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InputKind = Literal["number", "integer", "select"]


@dataclass(frozen=True)
class FeatureSpec:
    """Description of one model input feature."""

    column: str
    label: str
    help_text: str
    kind: InputKind = "number"
    unit: str | None = None
    options: tuple[tuple[float, str], ...] = ()
    step: float = 0.1
    format: str = "%.3f"

    @property
    def display_label(self) -> str:
        """Return the label shown in the Streamlit form."""
        if self.unit:
            return f"{self.label} ({self.unit})"
        return self.label


@dataclass(frozen=True)
class DiseaseSpec:
    """Configuration needed to render and run one classifier."""

    key: str
    title: str
    description: str
    model_filename: str
    dataset_filename: str
    target_column: str
    features: tuple[FeatureSpec, ...]
    positive_message: str
    negative_message: str


DIABETES_FEATURES = (
    FeatureSpec(
        "Pregnancies",
        "Pregnancies",
        "Number of times pregnant.",
        kind="integer",
        step=1.0,
        format="%d",
    ),
    FeatureSpec(
        "Glucose",
        "Glucose",
        "Plasma glucose concentration measured during the source dataset's glucose-tolerance test.",
        kind="number",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "BloodPressure",
        "Diastolic blood pressure",
        "Diastolic blood pressure measurement.",
        unit="mm Hg",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "SkinThickness",
        "Triceps skin-fold thickness",
        "Triceps skin-fold thickness used in the diabetes dataset.",
        unit="mm",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "Insulin",
        "2-hour serum insulin",
        "Serum insulin value recorded in the diabetes dataset.",
        unit="mu U/ml",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "BMI",
        "Body mass index",
        "Body mass index calculated from weight and height.",
        unit="kg/m^2",
        step=0.1,
        format="%.1f",
    ),
    FeatureSpec(
        "DiabetesPedigreeFunction",
        "Diabetes pedigree function",
        "Dataset-specific score summarizing diabetes-related family-history information.",
        step=0.001,
        format="%.3f",
    ),
    FeatureSpec(
        "Age",
        "Age",
        "Age in years.",
        kind="integer",
        unit="years",
        step=1.0,
        format="%d",
    ),
)


HEART_FEATURES = (
    FeatureSpec("age", "Age", "Age in years.", kind="integer", unit="years", step=1.0, format="%d"),
    FeatureSpec(
        "sex",
        "Sex code",
        "Encoding used by the dataset: 0 = female, 1 = male.",
        kind="select",
        options=((0, "0 - Female"), (1, "1 - Male")),
    ),
    FeatureSpec(
        "cp",
        "Chest-pain type",
        "The local CSV uses a 0-3 encoding of the four UCI chest-pain categories.",
        kind="select",
        options=(
            (0, "0 - Typical angina"),
            (1, "1 - Atypical angina"),
            (2, "2 - Non-anginal pain"),
            (3, "3 - Asymptomatic"),
        ),
    ),
    FeatureSpec(
        "trestbps",
        "Resting blood pressure",
        "Resting blood pressure recorded on admission.",
        unit="mm Hg",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "chol",
        "Serum cholesterol",
        "Serum cholesterol measurement.",
        unit="mg/dl",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "fbs",
        "Fasting blood sugar > 120 mg/dl",
        "Binary dataset field: 0 = false, 1 = true.",
        kind="select",
        options=((0, "0 - False"), (1, "1 - True")),
    ),
    FeatureSpec(
        "restecg",
        "Resting ECG result",
        (
            "UCI coding: 0 = normal, 1 = ST-T wave abnormality, "
            "2 = probable/definite left-ventricular hypertrophy."
        ),
        kind="select",
        options=(
            (0, "0 - Normal"),
            (1, "1 - ST-T wave abnormality"),
            (2, "2 - Left-ventricular hypertrophy"),
        ),
    ),
    FeatureSpec(
        "thalach",
        "Maximum heart rate achieved",
        "Maximum heart rate achieved during the source exercise test.",
        unit="bpm",
        step=1.0,
        format="%.1f",
    ),
    FeatureSpec(
        "exang",
        "Exercise-induced angina",
        "Binary dataset field: 0 = no, 1 = yes.",
        kind="select",
        options=((0, "0 - No"), (1, "1 - Yes")),
    ),
    FeatureSpec(
        "oldpeak",
        "ST depression (oldpeak)",
        "ST depression induced by exercise relative to rest.",
        step=0.1,
        format="%.1f",
    ),
    FeatureSpec(
        "slope",
        "Peak exercise ST-segment slope",
        "The local CSV uses 0-2 coding corresponding to upsloping, flat, and downsloping.",
        kind="select",
        options=((0, "0 - Upsloping"), (1, "1 - Flat"), (2, "2 - Downsloping")),
    ),
    FeatureSpec(
        "ca",
        "Major vessels (CA)",
        (
            "Number/code for major vessels colored by fluoroscopy. The local dataset "
            "contains values 0-4; value 4 is outside the original UCI 0-3 definition."
        ),
        kind="select",
        options=((0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4 - Present in local CSV")),
    ),
    FeatureSpec(
        "thal",
        "Thal code",
        (
            "Recoded thalassemia field used by the local CSV: 0 = unknown, "
            "1 = fixed defect, 2 = normal, 3 = reversible defect."
        ),
        kind="select",
        options=(
            (0, "0 - Unknown/not recorded"),
            (1, "1 - Fixed defect"),
            (2, "2 - Normal"),
            (3, "3 - Reversible defect"),
        ),
    ),
)


PARKINSONS_FEATURES = (
    FeatureSpec(
        "MDVP:Fo(Hz)",
        "Average vocal frequency",
        "Average vocal fundamental frequency.",
        unit="Hz",
    ),
    FeatureSpec(
        "MDVP:Fhi(Hz)",
        "Maximum vocal frequency",
        "Maximum vocal fundamental frequency.",
        unit="Hz",
    ),
    FeatureSpec(
        "MDVP:Flo(Hz)",
        "Minimum vocal frequency",
        "Minimum vocal fundamental frequency.",
        unit="Hz",
    ),
    FeatureSpec(
        "MDVP:Jitter(%)",
        "Jitter (%)",
        "Variation in fundamental frequency, expressed as a percentage.",
    ),
    FeatureSpec(
        "MDVP:Jitter(Abs)",
        "Absolute jitter",
        "Absolute variation in fundamental frequency.",
    ),
    FeatureSpec(
        "MDVP:RAP",
        "Jitter RAP",
        "Relative average perturbation measure of frequency variation.",
    ),
    FeatureSpec(
        "MDVP:PPQ",
        "Jitter PPQ",
        "Pitch perturbation quotient measure of frequency variation.",
    ),
    FeatureSpec("Jitter:DDP", "Jitter DDP", "Difference-of-differences period jitter measure."),
    FeatureSpec("MDVP:Shimmer", "Shimmer", "Variation in voice amplitude."),
    FeatureSpec(
        "MDVP:Shimmer(dB)",
        "Shimmer",
        "Amplitude variation measured in decibels.",
        unit="dB",
    ),
    FeatureSpec("Shimmer:APQ3", "Shimmer APQ3", "Three-point amplitude perturbation quotient."),
    FeatureSpec("Shimmer:APQ5", "Shimmer APQ5", "Five-point amplitude perturbation quotient."),
    FeatureSpec("MDVP:APQ", "Shimmer APQ", "MDVP amplitude perturbation quotient."),
    FeatureSpec("Shimmer:DDA", "Shimmer DDA", "Difference-of-differences amplitude measure."),
    FeatureSpec(
        "NHR",
        "Noise-to-harmonics ratio",
        "Ratio describing noise relative to tonal components.",
    ),
    FeatureSpec(
        "HNR",
        "Harmonics-to-noise ratio",
        "Ratio describing tonal components relative to noise.",
    ),
    FeatureSpec(
        "RPDE",
        "RPDE",
        "Recurrence period density entropy; a nonlinear dynamical complexity measure.",
    ),
    FeatureSpec("DFA", "DFA", "Detrended fluctuation analysis scaling exponent."),
    FeatureSpec("spread1", "Spread 1", "Nonlinear measure of fundamental-frequency variation."),
    FeatureSpec("spread2", "Spread 2", "Nonlinear measure of fundamental-frequency variation."),
    FeatureSpec("D2", "D2", "Nonlinear dynamical complexity measure."),
    FeatureSpec(
        "PPE",
        "PPE",
        "Pitch period entropy; a nonlinear measure of fundamental-frequency variation.",
    ),
)


DISEASE_SPECS: dict[str, DiseaseSpec] = {
    "Diabetes": DiseaseSpec(
        key="diabetes",
        title="Diabetes Classification",
        description=(
            "Eight-feature linear-SVM baseline trained on the diabetes CSV included "
            "with this project."
        ),
        model_filename="diabetes_model.sav",
        dataset_filename="diabetes.csv",
        target_column="Outcome",
        features=DIABETES_FEATURES,
        positive_message="Model output: diabetes-positive class.",
        negative_message="Model output: diabetes-negative class.",
    ),
    "Heart Disease": DiseaseSpec(
        key="heart_disease",
        title="Heart Disease Classification",
        description=(
            "Thirteen-feature logistic-regression baseline based on the Cleveland-style "
            "heart dataset."
        ),
        model_filename="heart_disease_model.sav",
        dataset_filename="heart.csv",
        target_column="target",
        features=HEART_FEATURES,
        positive_message="Model output: heart-disease-positive class.",
        negative_message="Model output: heart-disease-negative class.",
    ),
    "Parkinson's": DiseaseSpec(
        key="parkinsons",
        title="Parkinson's Disease Classification",
        description="Twenty-two-feature linear-SVM baseline using biomedical voice measurements.",
        model_filename="parkinsons_model.sav",
        dataset_filename="parkinsons.csv",
        target_column="status",
        features=PARKINSONS_FEATURES,
        positive_message="Model output: Parkinson's-positive class.",
        negative_message="Model output: Parkinson's-negative class.",
    ),
}


def feature_columns(spec: DiseaseSpec) -> tuple[str, ...]:
    """Return dataset feature columns in model input order."""
    return tuple(feature.column for feature in spec.features)
