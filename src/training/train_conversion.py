"""Stage 1: conversion model — will the brand onboard next quarter?

Candidate tournament under GroupKFold-by-quarter CV, champion selected
on PR-AUC (conversion is imbalanced and the cost asymmetry favors
precision: sales outreach is expensive). Every run is logged to MLflow
(sqlite backend); the champion plus its train-fit stats are persisted
to models/ as one joblib bundle.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:      # pragma: no cover
    HAS_LGBM = False

from src.features.build_features import TARGET
from src.features.splits import feature_columns, group_folds


def _make_candidate(name: str, params: dict):
    if name == "lgbm_classifier" and HAS_LGBM:
        return lgb.LGBMClassifier(random_state=42, verbosity=-1, **params)
    if name == "logreg_l2":
        return LogisticRegression(random_state=42, **params)
    if name == "gbt_classifier":
        return HistGradientBoostingClassifier(random_state=42, **params)
    raise ValueError(f"unknown conversion candidate: {name}")


def _cv_pr_auc(candidate, X: pd.DataFrame, y: pd.Series,
               quarters: pd.Series, n_folds: int) -> dict:
    """Out-of-fold PR-AUC / ROC-AUC with quarter-grouped folds."""
    import copy
    folds = group_folds(quarters, n_folds)
    oof = np.zeros(len(X))
    for f in sorted(set(folds)):
        tr, va = folds != f, folds == f
        m = copy.deepcopy(candidate)   # fresh fit per fold
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    return {"cv_pr_auc": float(average_precision_score(y, oof)),
            "cv_roc_auc": float(roc_auc_score(y, oof)),
            "oof_pred": oof}


def train_conversion(train: pd.DataFrame, cfg: dict, mlflow=None) -> dict:
    """Run the candidate tournament; return champion bundle (not yet fit on full train)."""
    tcfg = cfg["training"]["conversion"]
    feats = feature_columns(train)
    X, y = train[feats], train[TARGET].astype(int)
    quarters = train["snapshot_quarter"]
    n_folds = cfg["training"]["cv"]["folds"]

    results = []
    for name, spec in tcfg["candidates"].items():
        cand = _make_candidate(name, spec["params"])
        metrics = _cv_pr_auc(cand, X, y, quarters, n_folds)
        results.append({"name": name, **{k: v for k, v in metrics.items()
                                         if k != "oof_pred"}})
        if mlflow is not None:
            with mlflow.start_run(run_name=f"conversion/{name}", nested=True):
                mlflow.log_params(spec["params"])
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                    if k != "oof_pred"})

    results.sort(key=lambda r: r["cv_pr_auc"], reverse=True)
    champion_name = results[0]["name"]

    # Refit champion on the full training set for persistence.
    champion = _make_candidate(champion_name, tcfg["candidates"][champion_name]["params"])
    champion.fit(X, y)

    val_metrics = {}
    return {"model": champion, "name": champion_name, "features": feats,
            "tournament": results, "val_metrics": val_metrics}


def evaluate_conversion(model, valid: pd.DataFrame, feats: list[str]) -> dict:
    """Holdout metrics on the validation quarters."""
    Xv, yv = valid[feats], valid[TARGET].astype(int)
    p = model.predict_proba(Xv)[:, 1]
    return {"valid_pr_auc": float(average_precision_score(yv, p)),
            "valid_roc_auc": float(roc_auc_score(yv, p)),
            "valid_brier": float(brier_score_loss(yv, p)),
            "valid_base_rate": float(yv.mean())}


def save_bundle(bundle: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle["model"], "name": bundle["name"],
                 "features": bundle["features"],
                 "tournament": bundle["tournament"]}, path)
    return path
