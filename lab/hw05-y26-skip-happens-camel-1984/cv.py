import itertools

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from metrics import binary_log_loss
from model import Model
from selection import permutation_importance, select_features

SEED = 42
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)


def make_folds(X, y, n_splits=5, seeds=(SEED,)):
    return [list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
            for seed in seeds]


def oof_proba(X, y, folds, features, max_depth, min_samples_leaf, smoothing):
    oof = np.zeros(len(y))
    for train_idx, valid_idx in folds:
        model = Model("cv", features, max_depth=max_depth,
                      min_samples_leaf=min_samples_leaf, smoothing=smoothing)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])
    return oof


def best_threshold(y_true, proba, metric):
    scores = np.array([metric(y_true, (proba >= t).astype(int)) for t in THRESHOLDS])
    best = np.flatnonzero(scores == scores.max())
    i = best[len(best) // 2]
    return float(THRESHOLDS[i]), float(scores[i])


def fold_scores(y_true, proba, folds, metric, threshold):
    y_true = np.asarray(y_true)
    return np.array([metric(y_true[valid_idx], (proba[valid_idx] >= threshold).astype(int))
                     for _, valid_idx in folds])


def evaluate(X, y, fold_sets, scenarios, features, max_depth, min_samples_leaf, smoothing):
    n = len(y)
    oof = np.concatenate([oof_proba(X, y, folds, features, max_depth, min_samples_leaf, smoothing)
                          for folds in fold_sets])
    y_all = np.tile(np.asarray(y), len(fold_sets))
    all_folds = [(tr + i * n, va + i * n) for i, folds in enumerate(fold_sets) for tr, va in folds]
    row = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf,
           "smoothing": smoothing, "logloss": binary_log_loss(y_all, oof)}
    for case, metric in scenarios.items():
        threshold, score = best_threshold(y_all, oof, metric)
        row[f"{case}_thr"] = threshold
        row[f"{case}_score"] = score
        row[f"{case}_std"] = fold_scores(y_all, oof, all_folds, metric, threshold).std()
    return row


def grid_search(X, y, fold_sets, scenarios, features, grid):
    rows = [evaluate(X, y, fold_sets, scenarios, features, d, l, s)
            for d, l, s in itertools.product(grid["max_depth"], grid["min_samples_leaf"],
                                             grid["smoothing"])]
    return pd.DataFrame(rows)


def summarize(results, scenarios):
    rows = {case: results.loc[results[f"{case}_score"].idxmax()] for case in scenarios}
    rows["probability"] = results.loc[results["logloss"].idxmin()]
    return pd.DataFrame(rows).T


def cv_importance(X, y, fold_sets, features, params, score_fn, greater_is_better, n_repeats=5):
    total = dict.fromkeys(features, 0.0)
    folds = [pair for fs in fold_sets for pair in fs]
    for train_idx, valid_idx in folds:
        model = Model("cv", features, **params).fit(X.iloc[train_idx], y.iloc[train_idx])
        imp = permutation_importance(model, X.iloc[valid_idx][features], y.iloc[valid_idx],
                                     score_fn, greater_is_better, n_repeats=n_repeats, seed=SEED)
        for name, value in imp.items():
            total[name] += value / len(folds)
    return dict(sorted(total.items(), key=lambda kv: -kv[1]))


def selection_curve(X, y, fold_sets, scenarios, importances, params, ks):
    rows = []
    for k in ks:
        features = select_features(importances, k)
        row = evaluate(X, y, fold_sets, scenarios, features, params["max_depth"],
                       params["min_samples_leaf"], params["smoothing"])
        rows.append({"k": k, **row})
    return pd.DataFrame(rows)


def pick_simple(results, case, n_folds, logloss_tolerance=0.002):
    if case == "probability":
        best = results.loc[results["logloss"].idxmin()]
        close = results[results["logloss"] <= best["logloss"] + logloss_tolerance]
    else:
        col = f"{case}_score"
        best = results.loc[results[col].idxmax()]
        close = results[results[col] >= best[col] - best[f"{case}_std"] / np.sqrt(n_folds)]
    order = close.assign(depth=close["max_depth"].fillna(np.inf))
    return order.sort_values(["depth", "min_samples_leaf"], ascending=[True, False]).iloc[0]
