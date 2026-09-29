import pytest

from src.prediction import parse_numeric_inputs, predict_class


class DummyModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.last_input = None

    def predict(self, values):
        self.last_input = values
        return [self.prediction]


def test_parse_numeric_inputs_converts_values():
    assert parse_numeric_inputs(["1", "2.5", "-3"], 3) == [1.0, 2.5, -3.0]


def test_parse_numeric_inputs_rejects_missing_value():
    with pytest.raises(ValueError, match="required"):
        parse_numeric_inputs(["1", ""], 2)


def test_parse_numeric_inputs_rejects_non_numeric_value():
    with pytest.raises(ValueError, match="numeric"):
        parse_numeric_inputs(["1", "abc"], 2)


def test_parse_numeric_inputs_rejects_non_finite_value():
    with pytest.raises(ValueError, match="finite"):
        parse_numeric_inputs(["nan"], 1)


def test_parse_numeric_inputs_validates_feature_count():
    with pytest.raises(ValueError, match="Expected 2 values"):
        parse_numeric_inputs(["1"], 2)


def test_predict_class_calls_model_with_2d_input():
    model = DummyModel(1)
    result = predict_class(model, [1.0, 2.0])

    assert result == 1
    assert model.last_input == [[1.0, 2.0]]


def test_predict_class_rejects_non_binary_result():
    with pytest.raises(ValueError, match="binary class"):
        predict_class(DummyModel(2), [1.0])
