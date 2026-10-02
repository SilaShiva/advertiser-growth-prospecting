"""Stage 2: spend model — for brands that convert, expected first-90-day spend.

Trained ONLY on the positive cohort (converted rows). Target is
log1p(spend) so the heavy right tail doesn't dominate MSE/MAE; champion
is picked on MAE in log space. At inference the two stages compose as
expected_spend = P(convert) * expm1(spend_pred).
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:      # pragma: no cover
    HAS_LGBM = False

from src.features.splits import feature_columns, group_folds


def _make_candidate(name: str, params: dict):
    if name == "lgbm_regressor" and HAS_LGBM:
        return lgb.LGBMRegressor(random_state=42, verbosity=-1, **params)
    if name == "hist_gbr":
        return HistGradientBoostingRegressor(random_state=42, **params)
    raise ValueError(f"unknown spend candidate: {name}")


def _cv_mae_log1p(candidate, X: pd.DataFrame, y_log: pd.Series,
                  quarters: pd.Series, n_folds: int) -> dict:
    folds = group_folds(quarters, n_folds)
    oof = np.zeros(len(X))
    for f in sorted(set(folds)):
        tr, va = folds != f, folds == f
        m = copy.deepcopy(candidate)
        m.fit(X[tr], y_log[tr])
        oof[va] = m.predict(X[va])
    return {"cv_mae_log1p": float(mean_absolute_error(y_log, oof)),
            "oof_pred": oof}


def train_spend(train: pd.DataFrame, spend_labels: pd.DataFrame,
                cfg: dict, mlflow=None) -> dict:
    """Fit the spend model on converted training rows only."""
    tcfg = cfg["training"]["spend"]
    pos = train[train["converted_next_quarter"] == 1].copy()
    pos = pos.merge(spend_labels, on=["brand_id", "snapshot_quarter"], how="left")
    if pos["spend_90d"].isna().any():
        raise ValueError("converted rows missing spend labels after merge")
    feats = feature_columns(train)
    X = pos[feats]
    y_log = np.log1p(pos["spend_90d"])
    quarters = pos["snapshot_quarter"]
    n_folds = cfg["training"]["cv"]["folds"]

    results = []
    for name, spec in tcfg["candidates"].items():
        cand = _make_candidate(name, spec["params"])
        metrics = _cv_mae_log1p(cand, X, y_log, quarters, n_folds)
        results.append({"name": name, "cv_mae_log1p": metrics["cv_mae_log1p"]})
        if mlflow is not None:
            with mlflow.start_run(run_name=f"spend/{name}", nested=True):
                mlflow.log_params(spec["params"])
                mlflow.log_metrics({"cv_mae_log1p": metrics["cv_mae_log1p"]})

    results.sort(key=lambda r: r["cv_mae_log1p"])
    champion_name = results[0]["name"]

    champion = _make_candidate(champion_name, tcfg["candidates"][champion_name]["params"])
    champion.fit(X, y_log)

    return {"model": champion, "name": champion_name, "features": feats,
            "tournament": results}


def evaluate_spend(model, test: pd.DataFrame, spend_labels: pd.DataFrame,
                   feats: list[str]) -> dict:
    """Holdout metrics in INR (back-transformed) on the labeled test quarter."""
    pos = test[test["converted_next_quarter"] == 1].copy()
    if pos.empty:
        return {"test_spend_rows": 0}
    pos = pos.merge(spend_labels, on=["brand_id", "snapshot_quarter"], how="left")
    pred_inr = np.expm1(model.predict(pos[feats]))
    actual = pos["spend_90d"].to_numpy()
    return {
        "test_spend_rows": int(len(pos)),
        "test_mae_inr": float(mean_absolute_error(actual, pred_inr)),
        "test_mape_pct": float(np.mean(np.abs(pred_inr - actual) / actual) * 100),
        "test_r2": float(r2_score(actual, pred_inr)),
    }


def save_bundle(bundle: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle["model"], "name": bundle["name"],
                 "features": bundle["features"],
                 "tournament": bundle["tournament"]}, path)
    return path
