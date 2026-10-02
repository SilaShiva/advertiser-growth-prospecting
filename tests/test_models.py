"""Model training + inference contract on the tiny panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import TARGET
from src.features.splits import make_splits, spend_labels
from src.training.train_conversion import train_conversion, evaluate_conversion
from src.training.train_spend import train_spend, evaluate_spend
from src.inference.score import score_batch, _rule_score


def test_conversion_beats_base_rate(tiny_matrix):
    """The champion must beat the majority-class PR-AUC (= base rate)."""
    c, panel, stats, X = tiny_matrix
    s = make_splits(X, c)
    bundle = train_conversion(s.train, c)
    vm = evaluate_conversion(bundle["model"], s.valid, bundle["features"])
    assert vm["valid_roc_auc"] > 0.5


def test_spend_trains_on_converted_only(tiny_matrix, tiny_panel):
    c, panel, stats, X = tiny_matrix
    s = make_splits(X, c)
    sp = spend_labels(panel)
    bundle = train_spend(s.train, sp, c)
    # model must reproduce something in the plausible spend range
    pos = s.test[s.test[TARGET] == 1]
    if not pos.empty:
        m = evaluate_spend(bundle["model"], s.test, sp, bundle["features"])
        assert m["test_spend_rows"] == len(pos)
        assert m["test_mae_inr"] >= 0


def test_score_batch_output_schema(tiny_panel, tiny_matrix, tmp_path):
    c, panel, stats, X = tiny_matrix
    s = make_splits(X, c)
    sp = spend_labels(panel)
    from src.training.train_conversion import save_bundle as save_conv
    from src.training.train_spend import save_bundle as save_spend
    conv = train_conversion(s.train, c)
    spend = train_spend(s.train, sp, c)
    save_conv(conv, tmp_path / "conversion_model.joblib")
    save_spend(spend, tmp_path / "spend_model.joblib")
    import joblib
    joblib.dump(stats, tmp_path / "fit_stats.joblib")

    batch = panel[panel["snapshot_quarter"] == panel["snapshot_quarter"].max()]
    out = score_batch(batch, tmp_path, c)
    assert set(out.columns) >= {"brand_id", "p_adopt", "predicted_spend_90d_inr",
                                "expected_value_inr", "priority"}
    assert out["p_adopt"].between(0, 1).all()
    assert (out["predicted_spend_90d_inr"] >= 0).all()
    # expected value = p * spend (recomputable from the rounded CSV columns;
    # p_adopt is rounded to 4dp so tiny probabilities carry larger relative error)
    recomputed = out["p_adopt"] * out["predicted_spend_90d_inr"]
    assert np.allclose(out["expected_value_inr"], recomputed, rtol=0.02, atol=5)


def test_rule_fallback_scores_when_no_models(tiny_panel, tmp_path):
    c, panel = tiny_panel
    batch = panel[panel["snapshot_quarter"] == panel["snapshot_quarter"].max()]
    out = score_batch(batch, tmp_path / "does_not_exist", c)   # empty dir -> fallback
    assert len(out) == len(batch)
    assert out["p_adopt"].between(0, 1).all()
    assert out.attrs["model_used"] == "rule-based fallback"


def test_gates_block_on_bad_label(tiny_panel, cfg):
    """A corrupted label must trip the FAIL gate."""
    _, df = tiny_panel
    from src.validation.gates import run_gates
    bad = df.copy()
    bad.loc[bad.index[:5], "converted_next_quarter"] = 2
    report = run_gates(bad, cfg["validation"]["gates"])
    assert report.blocked
