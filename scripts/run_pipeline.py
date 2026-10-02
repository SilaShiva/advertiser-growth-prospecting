#!/usr/bin/env python
"""End-to-end orchestrator: generate -> gates -> features -> train -> evaluate.

Stops on any FAIL gate. Persists both champion bundles plus the train
fit-stats bundle that inference needs. MLflow (sqlite) logs every
candidate run; figures/metrics land in reports/ for review.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from src.config.loader import load_config
from src.ingestion.generate_data import generate_snapshots, write_snapshot_csv
from src.ingestion.load_raw import load_raw_panel
from src.validation.gates import run_gates, save_report
from src.features.build_features import fit_stats, build_features, TARGET
from src.features.splits import make_splits, spend_labels
from src.training.train_conversion import train_conversion, evaluate_conversion, save_bundle as save_conv
from src.training.train_spend import train_spend, evaluate_spend, save_bundle as save_spend
from src.evaluation.report import evaluate_champion


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-generation", action="store_true",
                    help="reuse existing data/raw CSVs")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    seed = cfg["project"]["random_seed"]
    reports = cfg["paths"]["reports_dir"]

    # ---- try MLflow (optional; pipeline works without it) ----
    mlflow = None
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{cfg['paths']['mlflow_dir'] / 'mlflow.db'}")
        mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
        mlflow = mlflow
        print(f"mlflow: tracking to {cfg['paths']['mlflow_dir'] / 'mlflow.db'}")
    except Exception as exc:                                   # pragma: no cover
        print(f"mlflow unavailable ({exc}); continuing without tracking")

    # ---- 1. data ----
    if args.skip_generation:
        print("1. data: reusing existing data/raw CSVs")
    else:
        print("1. data: generating synthetic snapshots ...")
        for old in cfg["paths"]["raw_dir"].glob("prospects_*.csv"):
            old.unlink()
        df_gen = generate_snapshots(cfg)
        write_snapshot_csv(df_gen, cfg["paths"]["raw_dir"],
                           cfg["generation"]["csv_float_fmt"])
    panel = load_raw_panel(cfg["paths"]["raw_dir"])
    print(f"   panel: {len(panel):,} rows x {len(panel.columns)} cols")

    # ---- 2. gates ----
    print("2. data-quality gates ...")
    report = run_gates(panel, cfg["validation"]["gates"])
    save_report(report, reports / "data_quality")
    for w in report.warnings:
        print(f"   WARN: {w}")
    if report.blocked:
        print("   FAIL gate hit — stopping. See reports/data_quality/gate_report.md")
        return 1
    print(f"   all gates passed ({len(report.results)} checks)")

    # ---- 3. features (train-only fit stats) ----
    print("3. features ...")
    quarters = sorted(panel["snapshot_quarter"].unique())
    train_q = quarters[:cfg["splits"]["train_quarters"]]
    stats = fit_stats(panel[panel["snapshot_quarter"].isin(train_q)], cfg)
    matrix = build_features(panel, stats, cfg)
    splits = make_splits(matrix, cfg)
    print(f"   matrix {matrix.shape[0]:,} x {matrix.shape[1]} | "
          f"train {len(splits.train):,} / valid {len(splits.valid):,} / "
          f"test {len(splits.test):,} / score {len(splits.score):,}")

    # ---- 4. train both stages ----
    print("4. training conversion tournament ...")
    conv = train_conversion(splits.train, cfg, mlflow=mlflow)
    for r in conv["tournament"]:
        print(f"   {r['name']:16} cv_pr_auc={r['cv_pr_auc']:.4f}")
    print(f"   champion: {conv['name']}")
    vm = evaluate_conversion(conv["model"], splits.valid, conv["features"])
    print(f"   valid: PR-AUC {vm['valid_pr_auc']:.4f} | ROC-AUC {vm['valid_roc_auc']:.4f}")

    print("5. training spend model (positive cohort) ...")
    sp_labels = spend_labels(panel)
    spend = train_spend(splits.train, sp_labels, cfg, mlflow=mlflow)
    for r in spend["tournament"]:
        print(f"   {r['name']:12} cv_mae_log1p={r['cv_mae_log1p']:.4f}")
    print(f"   champion: {spend['name']}")

    # ---- 5. persist artifacts ----
    models_dir = cfg["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    p1 = save_conv(conv, models_dir / "conversion_model.joblib")
    p2 = save_spend(spend, models_dir / "spend_model.joblib")
    p3 = models_dir / "fit_stats.joblib"
    joblib.dump(stats, p3)
    print(f"6. artifacts: {p1.name}, {p2.name}, {p3.name}")

    # ---- 6. evaluation on labeled test + score-batch projection ----
    print("7. evaluation ...")
    metrics = evaluate_champion(conv, spend, splits, sp_labels,
                                cfg["inference"]["conversion_threshold"],
                                reports / "metrics")
    print(f"   test PR-AUC {metrics['test_pr_auc']:.4f} | "
          f"ROC-AUC {metrics['test_roc_auc']:.4f} | "
          f"top-decile lift {metrics['test_top_decile_lift']:.2f}x")
    if "spend_mae_inr" in metrics:
        print(f"   spend MAE INR {metrics['spend_mae_inr']:,.0f}")

    # ---- 7. run summary ----
    summary = {
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
        "rows": len(panel),
        "features": len(conv["features"]),
        "champion_conversion": conv["name"],
        "champion_spend": spend["name"],
        "test_pr_auc": metrics["test_pr_auc"],
        "test_roc_auc": metrics["test_roc_auc"],
        "top_decile_lift": metrics["test_top_decile_lift"],
        "expected_pipeline_inr": metrics["score_batch"]["expected_pipeline_inr"],
    }
    (reports / "run_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(f"done in {summary['elapsed_sec']}s — summary in reports/run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
