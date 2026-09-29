from src.specs import DISEASE_SPECS


def test_expected_models_and_feature_counts():
    assert set(DISEASE_SPECS) == {"Diabetes", "Heart Disease", "Parkinson's"}
    assert len(DISEASE_SPECS["Diabetes"].features) == 8
    assert len(DISEASE_SPECS["Heart Disease"].features) == 13
    assert len(DISEASE_SPECS["Parkinson's"].features) == 22
