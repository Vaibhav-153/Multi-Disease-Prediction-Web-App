"""Feature metadata used by the Streamlit app and prediction validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseSpec:
    key: str
    title: str
    model_filename: str
    features: tuple[str, ...]
    positive_message: str
    negative_message: str


DISEASE_SPECS: dict[str, DiseaseSpec] = {
    "Diabetes": DiseaseSpec(
        key="diabetes",
        title="Diabetes Classification",
        model_filename="diabetes_model.sav",
        features=(
            "Pregnancies",
            "Glucose",
            "Blood Pressure",
            "Skin Thickness",
            "Insulin",
            "BMI",
            "Diabetes Pedigree Function",
            "Age",
        ),
        positive_message="Model output: diabetes-positive class.",
        negative_message="Model output: diabetes-negative class.",
    ),
    "Heart Disease": DiseaseSpec(
        key="heart_disease",
        title="Heart Disease Classification",
        model_filename="heart_disease_model.sav",
        features=(
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Serum Cholesterol",
            "Fasting Blood Sugar > 120 mg/dl",
            "Resting ECG Result",
            "Maximum Heart Rate",
            "Exercise-Induced Angina",
            "ST Depression (Oldpeak)",
            "Slope of Peak Exercise ST Segment",
            "Major Vessels (CA)",
            "Thal",
        ),
        positive_message="Model output: heart-disease-positive class.",
        negative_message="Model output: heart-disease-negative class.",
    ),
    "Parkinson's": DiseaseSpec(
        key="parkinsons",
        title="Parkinson's Disease Classification",
        model_filename="parkinsons_model.sav",
        features=(
            "MDVP:Fo(Hz)",
            "MDVP:Fhi(Hz)",
            "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)",
            "MDVP:Jitter(Abs)",
            "MDVP:RAP",
            "MDVP:PPQ",
            "Jitter:DDP",
            "MDVP:Shimmer",
            "MDVP:Shimmer(dB)",
            "Shimmer:APQ3",
            "Shimmer:APQ5",
            "MDVP:APQ",
            "Shimmer:DDA",
            "NHR",
            "HNR",
            "RPDE",
            "DFA",
            "Spread1",
            "Spread2",
            "D2",
            "PPE",
        ),
        positive_message="Model output: Parkinson's-positive class.",
        negative_message="Model output: Parkinson's-negative class.",
    ),
}
