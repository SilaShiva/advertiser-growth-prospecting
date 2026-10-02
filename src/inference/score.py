"""Inference: score the unlabeled production batch into an outreach list.

Contract: load both persisted bundles, rebuild features with the SAME
train-fit stats (stored alongside the bundles), score, and emit a
tiered outreach list:

- priority 1: P(adopt) >= conversion_threshold AND spend in top quartile
- priority 2: P(adopt) >= conversion_threshold
- priority 3: P(adopt) >= half the threshold (watchlist)

If artifacts are missing, a rule-based fallback score (config-weighted
composite of digital maturity, spend history, category index) keeps the
script useful before a first training run.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import TARGET


def _rule_score(df: pd.DataFrame, weights: dict) -> np.ndarray:
    """Deterministic heuristic when no trained bundle exists."""
    parts = [
        weights.get("digital_maturity", 0.3) * df["digital_maturity"].fillna(0),
        weights.get("dau_per_mau", 0.25) * (df["sessions_per_mau"].fillna(0).clip(0, 6) / 6),
        weights.get("category_gmv_index", 0.25) * (df["category_gmv_index"].fillna(0).clip(0, 1.5) / 1.5),
        weights.get("historical_ad_spend_log", 0.2)
        * (np.log1p(df["historical_ad_spend_inr"].fillna(0)) / 14),
    ]
    return np.clip(sum(parts), 0, 1)


def score_batch(score_df: pd.DataFrame, models_dir, cfg: dict) -> pd.DataFrame:
    """Return the outreach list sorted by expected value."""
    icfg = cfg["inference"]
    from pathlib import Path
    models_dir = Path(models_dir)

    conv_path = models_dir / "conversion_model.joblib"
    spend_path = models_dir / "spend_model.joblib"
    stats_path = models_dir / "fit_stats.joblib"

    if conv_path.exists() and spend_path.exists() and stats_path.exists():
        conv = joblib.load(conv_path)
        spend = joblib.load(spend_path)
        stats = joblib.load(stats_path)
        feats = conv["features"]

        # Rebuild features for the score batch with the persisted train stats.
        # build_features re-sorts rows, so ALL downstream columns must be
        # keyed off X (not score_df) to avoid row-order misalignment.
        from src.features.build_features import build_features
        X = build_features(score_df, stats, cfg)
        X = X[X["brand_id"].isin(score_df["brand_id"])].reset_index(drop=True)

        p = conv["model"].predict_proba(X[feats])[:, 1]
        spend_pred = np.expm1(spend["model"].predict(X[feats]))
        display = score_df.set_index("brand_id").loc[X["brand_id"]].reset_index()
        model_used = f"{conv['name']} + {spend['name']}"
    elif icfg.get("fallbacks", {}).get("enabled"):
        p = _rule_score(score_df, icfg["fallbacks"]["rule_score_weights"])
        spend_pred = np.full(len(score_df), 500_000.0)   # neutral prior
        display = score_df
        model_used = "rule-based fallback"
    else:
        raise FileNotFoundError(f"model bundles not found in {models_dir} "
                                "and fallbacks disabled")

    thr = icfg["conversion_threshold"]
    spend_thr = np.quantile(spend_pred, icfg["spend_percentile_tiers"][0])
    # Round first, then derive EV from the rounded columns so the CSV is
    # internally consistent (EV == p_adopt * spend, recomputable by readers).
    out = pd.DataFrame({
        "brand_id": display["brand_id"].to_numpy(),
        "brand_name": display["brand_name"].to_numpy(),
        "category": display["category"].to_numpy(),
        "snapshot_quarter": display["snapshot_quarter"].to_numpy(),
        "p_adopt": np.round(p, 4),
        "predicted_spend_90d_inr": np.round(spend_pred, 0),
    })
    out["expected_value_inr"] = np.round(
        out["p_adopt"] * out["predicted_spend_90d_inr"], 0)
    out["priority"] = np.where(
        (out["p_adopt"] >= thr) & (out["predicted_spend_90d_inr"] >= spend_thr), 1,
        np.where(out["p_adopt"] >= thr, 2,
                 np.where(out["p_adopt"] >= thr / 2, 3, 4)))
    out = out.sort_values(["priority", "expected_value_inr"],
                          ascending=[True, False]).reset_index(drop=True)
    out.attrs["model_used"] = model_used
    return out
