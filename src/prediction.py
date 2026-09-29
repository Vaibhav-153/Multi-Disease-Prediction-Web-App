"""Input validation and model inference helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def parse_numeric_inputs(values: Sequence[str], expected_count: int) -> list[float]:
    """Convert text inputs to finite floats and validate feature count."""
    if len(values) != expected_count:
        raise ValueError(f"Expected {expected_count} values, received {len(values)}.")

    parsed: list[float] = []
    for index, value in enumerate(values, start=1):
        text = value.strip()
        if not text:
            raise ValueError(f"Input {index} is required.")

        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"Input {index} must be numeric.") from exc

        if not math.isfinite(number):
            raise ValueError(f"Input {index} must be a finite number.")

        parsed.append(number)

    return parsed


def predict_class(model: Any, values: Sequence[float]) -> int:
    """Run one binary classification prediction and normalize the result."""
    result = model.predict([list(values)])
    if len(result) != 1:
        raise ValueError("Model returned an unexpected number of predictions.")

    predicted_class = int(result[0])
    if predicted_class not in {0, 1}:
        raise ValueError(f"Expected a binary class prediction, received {predicted_class}.")

    return predicted_class
