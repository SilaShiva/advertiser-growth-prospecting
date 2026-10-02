#!/usr/bin/env python
"""Score the newest (unlabeled) snapshot into a tiered outreach list.

Reads models/ produced by run_pipeline.py (or falls back to a
deterministic rule score if training hasn't run), writes
reports/scoring/outreach_list.csv + a summary to stdout.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loader import load_config
from src.ingestion.load_raw import load_raw_panel
from src.inference.score import score_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    panel = load_raw_panel(cfg["paths"]["raw_dir"])
    latest_q = panel["snapshot_quarter"].max()
    batch = panel[panel["snapshot_quarter"] == latest_q].copy()

    out = score_batch(batch, cfg["paths"]["models_dir"], cfg)

    scoring_dir = cfg["paths"]["reports_dir"] / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(scoring_dir / "outreach_list.csv", index=False)

    n = len(out)
    print(f"model: {out.attrs.get('model_used', 'unknown')}")
    print(f"scored {n} prospects from {latest_q} -> {scoring_dir / 'outreach_list.csv'}")
    for pr in (1, 2, 3, 4):
        cnt = int((out["priority"] == pr).sum())
        ev = out.loc[out["priority"] == pr, "expected_value_inr"].sum()
        print(f"  priority {pr}: {cnt:4d} brands | expected value INR {ev:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
