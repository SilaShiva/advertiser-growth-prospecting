"""Feature engineering: from raw panel to model matrix.

Everything is config-driven and leak-safe:

- log1p on heavy-tailed counts/spend columns;
- ratio features with guarded denominators (no division by zero);
- winsorization with quantiles fit on TRAIN only (passed in, never fit here);
- snapshot-over-snapshot deltas per brand (momentum signals);
- one-hot categories capped at top-N seen in train;
- median imputation with medians fit on TRAIN only.

The feature builder is a pure function of (panel, fit-stats, config):
at inference time the same function runs with train-fit stats, so the
production matrix cannot drift from the training matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DROP_COLS = [
    "brand_name", "first_contact_date", "snapshot_date", "_onboard_date",
    "spend_90d_inr",          # label 2, never a feature
]
TARGET = "converted_next_quarter"


def fit_stats(train: pd.DataFrame, cfg: dict) -> dict:
    """Fit all train-only statistics used downstream."""
    fcfg = cfg["features"]
    numeric = train.select_dtypes(include="number").drop(columns=[TARGET], errors="ignore")
    stats = {
        "medians": numeric.median().to_dict(),
        "winsor_lo": numeric.quantile(fcfg["winsorize_quantiles"][0]).to_dict(),
        "winsor_hi": numeric.quantile(fcfg["winsorize_quantiles"][1]).to_dict(),
        "top_categories": train["category"].value_counts()
                          .head(fcfg["categories_top_n"]).index.tolist(),
    }
    return stats


def _guarded_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den with zero-denominator guarded to NaN (imputed later)."""
    return np.where(den > 0, num / den.replace(0, np.nan), np.nan)


def build_features(panel: pd.DataFrame, stats: dict, cfg: dict) -> pd.DataFrame:
    """Build the model matrix from a raw panel using pre-fit stats."""
    fcfg = cfg["features"]
    df = panel.copy()

    # --- log1p on configured heavy-tailed columns ---
    for col in fcfg["log1p_cols"]:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col].clip(lower=0))

    # --- guarded ratios ---
    for ratio_name, num, den in fcfg["engagement_ratios"]:
        if num in df.columns and den in df.columns:
            df[ratio_name] = _guarded_ratio(df[num], df[den])

    # --- snapshot-over-snapshot momentum deltas per brand ---
    df = df.sort_values(["brand_id", "snapshot_date"]).reset_index(drop=True)
    g = df.groupby("brand_id")
    for w in fcfg["s2s_windows_quarters"]:
        for col in ("web_sessions_90d", "catalog_skus", "brand_gmv_share_pct",
                    "historical_ad_spend_inr"):
            if col in df.columns:
                prev = g[col].shift(w)
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[f"delta_{w}q_{col}"] = np.where(
                        prev > 0, (df[col] - prev) / prev, np.nan)

    # --- days since contact capped (older than 2 quarters saturates) ---
    df["days_since_contact_capped"] = df["days_since_contact"].clip(upper=180)

    # --- one-hot category with unseen -> "other" ---
    top = stats["top_categories"]
    df["category_bucket"] = np.where(df["category"].isin(top), df["category"], "other")
    cat_dummies = pd.get_dummies(df["category_bucket"], prefix="cat")
    df = pd.concat([df, cat_dummies], axis=1)

    # --- winsorize + impute numeric features with train stats ---
    feature_cols = [c for c in df.columns
                    if c not in DROP_COLS + [TARGET, "category", "category_bucket"]
                    and not c.startswith("snapshot")]
    for col in feature_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lo = stats["winsor_lo"].get(col)
            hi = stats["winsor_hi"].get(col)
            if lo is not None and hi is not None:
                df[col] = df[col].clip(lo, hi)
            med = stats["medians"].get(col)
            df[col] = df[col].fillna(med if med is not None else 0.0)

    keep = feature_cols + ["brand_id", "snapshot_quarter", TARGET]
    keep = [c for c in keep if c in df.columns]
    return df[keep]
