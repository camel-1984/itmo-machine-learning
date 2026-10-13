"""Показатель и его параметры выбирает студент, независимо от сценария."""

import numbers

import numpy as np

def permutation_importance(model, X, y, score_fn, greater_is_better,
                           n_repeats=3, seed=42):
    """score_fn(model, X, y) возвращает скаляр с фиксированными настройками.

    Importance положительна при ухудшении качества после перестановки.
    Модель не переобучается внутри измерения importance.
    Вернуть словарь {имя колонки X: средняя importance по перестановкам}.
    greater_is_better задаёт направление улучшения score_fn.
    Для показателя, где больше лучше: importance = исходный score - score
    после перестановки; где меньше лучше — разность с обратным знаком.
    n_repeats — положительное целое число; seed обеспечивает воспроизводимость.
    """
    if isinstance(n_repeats, bool) or not isinstance(n_repeats, numbers.Integral) or n_repeats < 1:
        raise ValueError("n_repeats must be a positive integer")

    rng = np.random.default_rng(seed)
    sign = 1.0 if greater_is_better else -1.0
    base = score_fn(model, X, y)
    X_perm = X.copy()
    importances = {}
    for col in X.columns:
        values = X[col].to_numpy()
        drops = []
        for _ in range(n_repeats):
            X_perm[col] = rng.permutation(values)
            drops.append(sign * (base - score_fn(model, X_perm, y)))
        X_perm[col] = values
        importances[col] = float(np.mean(drops))
    return importances


def select_features(importances, n_features):
    """Вернуть имена n_features признаков с наибольшей importance.

    importances — словарь {имя признака: числовая importance}.
    n_features — целое число от 1 до размера словаря; иначе ValueError.
    При равной importance сортировать имена по алфавиту.
    Размер набора студент выбирает по локальной валидации.
    """
    if isinstance(n_features, bool) or not isinstance(n_features, numbers.Integral):
        raise ValueError("n_features must be an integer")
    if not 1 <= n_features <= len(importances):
        raise ValueError("n_features must be between 1 and the number of features")

    ranked = sorted(importances, key=lambda name: (-importances[name], name))
    return ranked[:n_features]

