#!/usr/bin/env python
"""Regenerate the synthetic prospect dataset into data/raw/.

Deterministic (fixed seed in config). Safe to run any time; output
replaces the previous CSVs byte-for-byte. No real data involved.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.loader import load_config
from src.ingestion.generate_data import generate_snapshots, write_snapshot_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg["paths"]["raw_dir"]

    # Clear stale files first so removed quarters don't linger.
    for old in sorted(raw_dir.glob("prospects_*.csv")):
        old.unlink()

    df = generate_snapshots(cfg)
    written = write_snapshot_csv(df, raw_dir, cfg["generation"]["csv_float_fmt"])
    conv = df["converted_next_quarter"].mean()
    print(f"generated {len(df):,} rows x {len(df.columns)} cols "
          f"-> {len(written)} quarterly CSVs in {raw_dir}")
    print(f"conversion rate: {conv:.3f} | "
          f"converted rows: {int(df['converted_next_quarter'].sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
