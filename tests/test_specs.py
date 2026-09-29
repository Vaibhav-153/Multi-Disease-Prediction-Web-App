from src.specs import DISEASE_SPECS, feature_columns


def test_expected_feature_counts() -> None:
    assert len(DISEASE_SPECS["Diabetes"].features) == 8
    assert len(DISEASE_SPECS["Heart Disease"].features) == 13
    assert len(DISEASE_SPECS["Parkinson's"].features) == 22


def test_feature_columns_are_unique_and_documented() -> None:
    for spec in DISEASE_SPECS.values():
        columns = feature_columns(spec)
        assert len(columns) == len(set(columns))
        assert all(feature.help_text.strip() for feature in spec.features)
        assert all(feature.label.strip() for feature in spec.features)


def test_categorical_heart_fields_have_explicit_options() -> None:
    heart = DISEASE_SPECS["Heart Disease"]
    select_features = {
        feature.column: feature for feature in heart.features if feature.kind == "select"
    }

    assert set(select_features) == {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
    assert all(feature.options for feature in select_features.values())


def test_model_and_dataset_filenames_are_local_names() -> None:
    for spec in DISEASE_SPECS.values():
        assert "/" not in spec.model_filename
        assert "\\" not in spec.model_filename
        assert spec.model_filename.endswith(".sav")
        assert spec.dataset_filename.endswith(".csv")
