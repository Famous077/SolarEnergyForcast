"""Weeks 6-8: LightGBM direct multi-horizon models with P10/P50/P90 quantiles.

One model per (site, horizon, quantile). Hyperparameters are tuned once per (site, horizon)
with Optuna on rolling-origin folds inside the training set, then shared by all three quantiles.
Training and scoring use daytime hours only.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build_features, chronological_split, daytime_trainable, rolling_folds

QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
FIXED = {"objective": "quantile", "bagging_freq": 1, "verbose": -1, "n_jobs": -1}
MAX_TREES = 2000
EARLY_STOP = 50


def _xy(X: pd.DataFrame):
    return X.drop(columns="y"), X["y"]


def _fit(train: pd.DataFrame, alpha: float, params: dict, valid: pd.DataFrame | None = None):
    Xt, yt = _xy(train)
    model = lgb.LGBMRegressor(**{"n_estimators": MAX_TREES, **FIXED, **params}, alpha=alpha)
    if valid is None:
        return model.fit(Xt, yt)
    Xv, yv = _xy(valid)
    return model.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric="l2",
                     callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True, verbose=False)])


def tune(train: pd.DataFrame, n_trials: int = 25, n_splits: int = 4, seed: int = 0) -> dict:
    """Search P50 params minimising mean daytime RMSE over rolling folds.
    `train` is the full hourly feature frame (nights included) so the fold gap is in hours.
    Returns the best params, with n_estimators = mean early-stopping round of the best trial."""
    import optuna      # only needed for training, so the deployed app doesn't have to install it
    folds = [(daytime_trainable(train.iloc[a]), daytime_trainable(train.iloc[b]))
             for a, b in rolling_folds(train, n_splits=n_splits)]

    def objective(trial):
        params = {
            "n_estimators": MAX_TREES,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        rmses, rounds = [], []
        for tr, va in folds:
            m = _fit(tr, 0.5, params, valid=va)
            rmses.append(np.sqrt(np.mean((m.predict(_xy(va)[0]) - va["y"]) ** 2)))
            rounds.append(m.best_iteration_)
        trial.set_user_attr("n_estimators", int(np.mean(rounds)))
        return float(np.mean(rmses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "n_estimators": study.best_trial.user_attrs["n_estimators"],
            "cv_rmse": study.best_value}


def fit_quantiles(train: pd.DataFrame, params: dict) -> dict:
    """Fit P10/P50/P90 on all daytime training rows."""
    p = {k: v for k, v in params.items() if k != "cv_rmse"}
    day = daytime_trainable(train)
    return {name: _fit(day, a, p) for name, a in QUANTILES.items()}


def train_from_params(df: pd.DataFrame, horizon: int, mode: str, params: dict):
    """Refit the three quantile models from tuned params (saved by notebook 05).
    Returns (models, X) where X is the full feature frame (train + test)."""
    X = build_features(df, horizon, weather_mode=mode)
    train, _ = chronological_split(X)
    return fit_quantiles(train, params), X


def predict_quantiles(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Predict every row; sort quantiles so they never cross, clip to the CF range, zero at night."""
    feats = X.drop(columns="y")
    raw = np.column_stack([models[k].predict(feats) for k in QUANTILES])
    out = pd.DataFrame(np.sort(raw, axis=1).clip(0, 1.05), index=X.index, columns=list(QUANTILES))
    out.loc[X["is_day"] == 0] = 0.0
    return out
