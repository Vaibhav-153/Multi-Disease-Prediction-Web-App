"""Streamlit interface for the three disease-classification models."""

from __future__ import annotations

import streamlit as st

from src.model_io import load_model
from src.prediction import parse_numeric_inputs, predict_class
from src.specs import DISEASE_SPECS, DiseaseSpec

st.set_page_config(
    page_title="Multi-Disease Prediction Demo",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def get_model(filename: str):
    return load_model(filename)


def render_prediction_page(spec: DiseaseSpec) -> None:
    st.title(spec.title)
    st.caption(
        "Educational machine-learning demo. The result is a model classification, "
        "not a medical diagnosis or clinical recommendation."
    )

    raw_values: list[str] = []
    columns = st.columns(3)

    for index, feature_name in enumerate(spec.features):
        with columns[index % 3]:
            raw_values.append(
                st.text_input(
                    feature_name,
                    key=f"{spec.key}_{index}",
                    placeholder="Enter a numeric value",
                )
            )

    if st.button("Run prediction", type="primary", key=f"predict_{spec.key}"):
        try:
            values = parse_numeric_inputs(raw_values, len(spec.features))
            model = get_model(spec.model_filename)
            predicted_class = predict_class(model, values)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return
        except Exception:
            st.error("Prediction failed. Check the model artifact and input values.")
            return

        message = spec.positive_message if predicted_class == 1 else spec.negative_message
        st.success(message)
        st.info(
            "Do not use this output to make medical decisions. "
            "Consult a qualified healthcare professional for diagnosis or treatment."
        )


def main() -> None:
    st.sidebar.title("Prediction Menu")
    selected = st.sidebar.radio("Choose a model", list(DISEASE_SPECS))

    st.sidebar.markdown("---")
    st.sidebar.caption("Models: linear SVM / logistic regression")

    render_prediction_page(DISEASE_SPECS[selected])


if __name__ == "__main__":
    main()
