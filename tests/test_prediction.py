import pytest

from src.prediction import parse_numeric_inputs, predict_class


class FakeModel:
    n_features_in_ = 2

    def __init__(self, result):
        self.result = result

    def predict(self, values):
        assert values == [[1.0, 2.0]]
        return self.result


def test_parse_numeric_inputs() -> None:
    assert parse_numeric_inputs(["1", "2.5"], 2) == [1.0, 2.5]


@pytest.mark.parametrize("value", ["", "   "])
def test_parse_numeric_inputs_rejects_blank(value: str) -> None:
    with pytest.raises(ValueError, match="required"):
        parse_numeric_inputs([value], 1)


def test_parse_numeric_inputs_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="numeric"):
        parse_numeric_inputs(["abc"], 1)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_parse_numeric_inputs_rejects_non_finite(value: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        parse_numeric_inputs([value], 1)


def test_parse_numeric_inputs_rejects_wrong_count() -> None:
    with pytest.raises(ValueError, match="Expected 2 values"):
        parse_numeric_inputs(["1"], 2)


def test_predict_class_returns_binary_value() -> None:
    assert predict_class(FakeModel([1]), [1.0, 2.0]) == 1


def test_predict_class_checks_model_feature_count() -> None:
    with pytest.raises(ValueError, match="expects 2 features"):
        predict_class(FakeModel([1]), [1.0])


def test_predict_class_rejects_non_binary_result() -> None:
    with pytest.raises(ValueError, match="binary class"):
        predict_class(FakeModel([2]), [1.0, 2.0])
