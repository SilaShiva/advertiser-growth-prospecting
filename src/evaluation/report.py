"""Evaluation: holdout metrics, gain curves, calibration, figures.

Produces the committed evidence in reports/:
- metrics/model_metrics.json        — every number in one place
- metrics/model_report.md           — human-readable summary
- figures/gain_curve.png            — cumulative conversions by ranked decile
- figures/calibration.png           — predicted vs observed by probability bin
- figures/spend_scatter.png         — predicted vs actual spend (test converters)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

from src.features.build_features import TARGET


def _decile_gain_table(y: pd.Series, p: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"y": y.to_numpy(), "p": p})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=False,
                           duplicates="drop")
    df["decile"] = 9 - df["decile"]          # decile 1 = highest scores
    g = df.groupby("decile").agg(brands=("y", "size"), conversions=("y", "sum"))
    g["conv_rate"] = g["conversions"] / g["brands"]
    g["cum_conversions"] = g["conversions"].cumsum()
    g["cum_capture_pct"] = g["cum_conversions"] / g["conversions"].sum()
    return g


def _fig_gain(g: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(g) + 1), g["cum_capture_pct"] * 100, marker="o")
    ax.plot([1, len(g)], [0, 100], linestyle="--", color="gray", linewidth=1,
            label="random targeting")
    ax.set_xlabel("prospect decile (1 = highest P(adopt))")
    ax.set_ylabel("cumulative % of conversions captured")
    ax.set_title("Targeting gain curve — holdout quarter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_calibration(y: pd.Series, p: np.ndarray, out: Path, n_bins: int = 8) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label="model")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect")
    ax.set_xlabel("mean predicted P(adopt)")
    ax.set_ylabel("observed conversion rate")
    ax.set_title("Calibration — holdout quarter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_spend_scatter(actual: np.ndarray, pred_inr: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    lim = max(actual.max(), pred_inr.max()) * 1.05
    ax.scatter(actual, pred_inr, alpha=0.7, edgecolor="k", linewidth=0.3)
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1)
    ax.set_xlabel("actual first-90d spend (INR)")
    ax.set_ylabel("predicted spend (INR)")
    ax.set_title("Spend model — labeled test quarter")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def evaluate_champion(model_conv, model_spend, splits, spend_labels_df,
                      threshold: float, out_dir: Path) -> dict:
    """Full evaluation on the labeled holdout quarter; writes reports."""
    out_dir = Path(out_dir)
    test = splits.test
    feats = model_conv["features"]
    p = model_conv["model"].predict_proba(test[feats])[:, 1]
    y = test[TARGET].astype(int)

    metrics = {
        "champion_conversion": model_conv["name"],
        "champion_spend": model_spend["name"],
        "conversion_tournament": model_conv["tournament"],
        "spend_tournament": model_spend["tournament"],
        "test_rows": int(len(test)),
        "test_base_rate": float(y.mean()),
        "test_pr_auc": float(average_precision_score(y, p)),
        "test_roc_auc": float(roc_auc_score(y, p)),
        "test_brier": float(brier_score_loss(y, p)),
        "test_precision_at_threshold": float(precision_score(y, p >= threshold, zero_division=0)),
        "test_recall_at_threshold": float(recall_score(y, p >= threshold, zero_division=0)),
        "threshold": threshold,
    }

    # Top-decile lift: the number leadership asks for.
    gain = _decile_gain_table(y, p)
    top = gain.iloc[0]
    metrics["test_top_decile_lift"] = float(top["conv_rate"] / y.mean())
    metrics["test_top_decile_capture_pct"] = float(top["cum_capture_pct"] * 10)
    metrics["decile_gain"] = gain.reset_index().to_dict(orient="records")

    # Spend model on test converters.
    pos = test[test[TARGET] == 1].copy()
    if not pos.empty:
        pos = pos.merge(spend_labels_df, on=["brand_id", "snapshot_quarter"], how="left")
        pred_inr = np.expm1(model_spend["model"].predict(pos[feats]))
        actual = pos["spend_90d"].to_numpy()
        metrics["spend_test_rows"] = int(len(pos))
        metrics["spend_mae_inr"] = float(np.mean(np.abs(pred_inr - actual)))
        metrics["spend_mape_pct"] = float(np.mean(np.abs(pred_inr - actual) / actual) * 100)
        metrics["spend_r2"] = float(np.corrcoef(actual, pred_inr)[0, 1] ** 2)
        _fig_spend_scatter(actual, pred_inr, out_dir.parent / "figures" / "spend_scatter.png")

    # Expected-pipeline value on the unlabeled score batch.
    score = splits.score
    p_score = model_conv["model"].predict_proba(score[feats])[:, 1]
    spend_score = np.expm1(model_spend["model"].predict(score[feats]))
    metrics["score_batch"] = {
        "rows": int(len(score)),
        "expected_converters": float((p_score >= threshold).sum()),
        "expected_pipeline_inr": float((p_score * spend_score).sum()),
        "mean_p_convert": float(p_score.mean()),
    }

    (out_dir / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float), encoding="utf-8")

    _fig_gain(gain, out_dir.parent / "figures" / "gain_curve.png")
    _fig_calibration(y, p, out_dir.parent / "figures" / "calibration.png")

    return metrics
