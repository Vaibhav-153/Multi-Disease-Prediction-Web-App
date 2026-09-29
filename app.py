"""Streamlit interface for the three disease-classification models."""

from __future__ import annotations

import streamlit as st

from src.model_io import load_model
from src.prediction import parse_numeric_inputs, predict_class
from src.specs import DISEASE_SPECS, DiseaseSpec, FeatureSpec

st.set_page_config(
    page_title="Multi-Disease Prediction Demo",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def get_model(filename: str):
    """Load and cache a trusted local model artifact."""
    return load_model(filename)


def render_feature_input(feature: FeatureSpec, widget_key: str) -> str:
    """Render one configured input and return a string for shared validation."""
    if feature.kind == "select":
        labels = {value: label for value, label in feature.options}
        selected = st.selectbox(
            feature.display_label,
            options=[None, *labels],
            index=0,
            format_func=lambda value: "Select a value" if value is None else labels[value],
            help=feature.help_text,
            key=widget_key,
        )
        return "" if selected is None else str(selected)

    step: int | float
    if feature.kind == "integer":
        step = max(1, int(feature.step))
    else:
        step = float(feature.step)

    value = st.number_input(
        feature.display_label,
        value=None,
        step=step,
        format=feature.format,
        help=feature.help_text,
        key=widget_key,
    )
    return "" if value is None else str(value)


def render_field_guide(spec: DiseaseSpec) -> None:
    """Show concise input documentation inside the app."""
    with st.expander("Input field guide"):
        st.write(
            "Values must use the same definitions and encodings as the training dataset. "
            "Categorical fields are provided as drop-downs where the dataset uses coded values."
        )
        for feature in spec.features:
            unit = f" [{feature.unit}]" if feature.unit else ""
            st.markdown(f"**{feature.label}{unit}** — {feature.help_text}")


def render_prediction_page(spec: DiseaseSpec) -> None:
    """Render one classifier page and its prediction action."""
    st.title(spec.title)
    st.write(spec.description)
    st.caption(
        "Educational machine-learning demo. The result is a model classification, "
        "not a medical diagnosis or clinical recommendation."
    )

    render_field_guide(spec)

    raw_values: list[str] = []
    columns = st.columns(3)
    for index, feature in enumerate(spec.features):
        with columns[index % 3]:
            raw_values.append(
                render_feature_input(feature, widget_key=f"{spec.key}_{feature.column}_{index}")
            )

    st.caption(f"Required inputs: {len(spec.features)}")

    if st.button("Run prediction", type="primary", key=f"predict_{spec.key}"):
        try:
            values = parse_numeric_inputs(raw_values, len(spec.features))
            model = get_model(spec.model_filename)
            predicted_class = predict_class(model, values)
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
            return
        except Exception:
            st.error(
                "Prediction failed. Check that the model artifact matches the installed "
                "scikit-learn version and that all required project files are present."
            )
            return

        message = spec.positive_message if predicted_class == 1 else spec.negative_message
        st.success(message)
        st.info(
            "Do not use this output to make medical decisions. "
            "Consult a qualified healthcare professional for diagnosis or treatment."
        )


def main() -> None:
    """Run the Streamlit application."""
    st.sidebar.title("Prediction Menu")
    selected = st.sidebar.radio("Choose a model", list(DISEASE_SPECS))
    st.sidebar.markdown("---")
    st.sidebar.caption("Baseline models: linear SVM / logistic regression")
    st.sidebar.caption("Portfolio and learning project — not for clinical use")

    render_prediction_page(DISEASE_SPECS[selected])


if __name__ == "__main__":
    main()
