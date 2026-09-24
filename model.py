"""The model. One pooled random forest, out-of-fold predictions grouped by player."""

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict

N_SPLITS = 5

# Settled by the tuning runs. Depth 5 is deliberately absent: it won an earlier
# grid and flattened the top of the distribution.
PARAMS = dict(n_estimators=400, max_depth=10, max_features=0.5,
              min_samples_leaf=4, random_state=42)

GRID = {'n_estimators': [200, 400],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5]}


def make(**overrides):
    return RandomForestRegressor(**{**PARAMS, **overrides})


def oof(X, y, groups, model=None, n_splits=N_SPLITS):
    """Out-of-fold predictions for every row. GroupKFold on `code`, so no
    player is ever scored by a model that trained on him."""
    model = make() if model is None else model
    return cross_val_predict(clone(model), X, y, groups=groups,
                             cv=GroupKFold(n_splits=n_splits), n_jobs=-1)


def tune(X, y, groups, grid=None, n_splits=N_SPLITS, scoring='neg_mean_absolute_error'):
    """Grid search under the same grouped folds. MAE on a percentile target is
    a ranking-ish loss and is not dominated by the tail."""
    gs = GridSearchCV(RandomForestRegressor(random_state=42),
                      grid or GRID, cv=GroupKFold(n_splits=n_splits),
                      scoring=scoring, n_jobs=-1)
    gs.fit(X, y, groups=groups)
    return gs


def importances(X, y, model=None):
    import pandas as pd
    model = make() if model is None else model
    model.fit(X, y)
    return pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)