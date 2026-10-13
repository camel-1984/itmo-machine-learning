"""Реализуйте метрики самостоятельно. Положительный класс — skip=1.

Входы: непустые одномерные массивы одинаковой длины. Метки только 0/1.
Для вероятностей допустимы конечные значения в [0, 1].
Некорректные входы отклоняются с ValueError.
Каждая функция возвращает скаляр float.
"""

import numpy as np

def _check_inputs(
        y_true,
        y_pred,
        probabilities = False
) -> tuple[np.ndarray, np.ndarray]:
    try:
        y_true_nd = np.asarray(y_true, dtype=float)
        y_pred_nd = np.asarray(y_pred, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("inputs must be numeric")
    if (
            y_true_nd.size == 0 or
            y_pred_nd.size == 0 or
            y_true_nd.ndim != 1 or
            y_pred_nd.ndim != 1 or
            y_true_nd.size != y_pred_nd.size):
        raise ValueError("inputs must be non-empty 1-D arrays of the same length")
    if not np.isin(y_true_nd, [0, 1]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if probabilities:
        if not (np.isfinite(y_pred_nd).all() and
                (y_pred_nd >= 0).all() and
                (y_pred_nd <= 1).all()):
            raise ValueError("probabilities must be finite and in [0, 1]")
    elif not np.isin(y_pred_nd, [0, 1]).all():
        raise ValueError("y_pred must contain only 0 and 1")
    return y_true_nd, y_pred_nd

def _check_param(value, name) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value

def _confusion(
        y_true,
        y_pred
) -> tuple[float, float, float]:

    tp = (y_true * y_pred).sum()
    fp = (y_pred == 1).sum() - tp
    fn = (y_true == 1).sum() - tp

    return tp, fp, fn


def precision(y_true, y_pred):
    """TP/(TP+FP). Нулевой знаменатель даёт 0."""

    y_true, y_pred = _check_inputs(y_true, y_pred)
    tp, fp, _ = _confusion(y_true, y_pred)

    return float(tp / (tp + fp)) if tp + fp != 0 else 0.0


def recall(y_true, y_pred):
    """TP/(TP+FN). Нулевой знаменатель даёт 0."""

    y_true, y_pred = _check_inputs(y_true, y_pred)
    tp, _, fn = _confusion(y_true, y_pred)

    return float(tp / (tp + fn)) if tp + fn != 0 else 0.0



def fbeta(y_true, y_pred, beta=1.0):
    """(1+beta**2)*TP/((1+beta**2)*TP+beta**2*FN+FP).
    beta — конечное положительное число. Нулевой знаменатель даёт 0.
    """
    beta = _check_param(beta, "beta")
    if beta <= 0:
        raise ValueError("beta must be above zero")

    y_true, y_pred = _check_inputs(y_true, y_pred)
    tp, fp, fn = _confusion(y_true, y_pred)


    return float((1+beta**2)*tp/((1+beta**2)*tp+beta**2*fn+fp)) \
        if ((1+beta**2)*tp+beta**2*fn+fp) != 0 else 0.0



def error_cost(y_true, y_pred, fp_cost=1.0, fn_cost=1.0):
    """(fp_cost*FP + fn_cost*FN)/N.

    Веса — конечные неотрицательные числа.
    """
    fp_cost = _check_param(fp_cost, "fp_cost")
    fn_cost = _check_param(fn_cost, "fn_cost")
    if fp_cost < 0 or fn_cost < 0:
        raise ValueError("cost must be non-negative")

    y_true, y_pred = _check_inputs(y_true, y_pred)
    tp, fp, fn = _confusion(y_true, y_pred)

    return float((fp_cost*fp + fn_cost*fn)/y_true.size)


def binary_log_loss(y_true, probabilities, eps=1e-6):
    """Среднее -y*log(p)-(1-y)*log(1-p), натуральный логарифм.

    eps — конечное число в (0, 0.5).
    Допустимые p обрезаются в [eps, 1-eps].
    """

    eps = _check_param(eps, "eps")
    if not (0 < eps < 0.5):
        raise ValueError("eps must be in (0, 0.5)")

    y_true, p = _check_inputs(y_true, probabilities, True)
    p = np.clip(p, eps, 1 - eps)

    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
