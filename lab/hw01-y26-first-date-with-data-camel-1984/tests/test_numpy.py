import ast
import inspect

import numpy as np
import pytest

from solutions import prepare_price_series, detect_price_jumps


def test_prepare_price_series_basic():
    prices = np.array([100.0, np.nan, 120.0, -5.0, np.inf, 90.0])
    original = prices.copy()

    clean, changes = prepare_price_series(prices)

    np.testing.assert_array_equal(clean, np.array([100.0, 120.0, 90.0]))
    np.testing.assert_allclose(changes, np.array([0.2, -0.25]))
    np.testing.assert_array_equal(prices, original)


def test_prepare_price_series_short_result():
    clean, changes = prepare_price_series(
        np.array([np.nan, -2.0, 15.0, np.inf])
    )

    np.testing.assert_array_equal(clean, np.array([15.0]))
    assert isinstance(changes, np.ndarray)
    assert changes.size == 0


def test_detect_price_jumps_uses_original_indices():
    prices = np.array([100.0, np.nan, 110.0, -1.0, 165.0])

    result = detect_price_jumps(prices, threshold=0.4)

    np.testing.assert_array_equal(result, np.array([4]))
    assert np.issubdtype(result.dtype, np.integer)


def test_detect_price_jumps_negative_threshold():
    with pytest.raises(ValueError):
        detect_price_jumps(np.array([100.0, 120.0]), threshold=-0.1)


@pytest.mark.parametrize("func", [prepare_price_series, detect_price_jumps])
def test_numpy_tasks_are_vectorized(func):
    tree = ast.parse(inspect.getsource(func))
    forbidden = (ast.For, ast.While, ast.ListComp)
    assert not any(isinstance(node, forbidden) for node in ast.walk(tree))
