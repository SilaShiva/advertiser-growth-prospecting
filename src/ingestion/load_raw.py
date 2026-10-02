"""Load the raw quarterly CSVs into one panel DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_panel(raw_dir: Path) -> pd.DataFrame:
    """Concatenate all prospects_*.csv files, parsing dates."""
    files = sorted(Path(raw_dir).glob("prospects_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no prospects_*.csv under {raw_dir} — run scripts/generate_data.py first")
    frames = [pd.read_csv(f, parse_dates=["snapshot_date", "first_contact_date"])
              for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df
