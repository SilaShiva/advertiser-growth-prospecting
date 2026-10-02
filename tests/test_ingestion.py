"""Generator determinism + schema invariants."""
from __future__ import annotations

import pandas as pd

from src.ingestion.generate_data import generate_snapshots


def test_generator_is_deterministic(tiny_panel):
    c, df1 = tiny_panel
    df2 = generate_snapshots(c)
    pd.testing.assert_frame_equal(df1, df2)


def test_panel_schema(tiny_panel):
    _, df = tiny_panel
    assert {"brand_id", "snapshot_quarter", "converted_next_quarter",
            "spend_90d_inr"} <= set(df.columns)
    # labels are 0/1 only; spend present only for converters
    assert set(df["converted_next_quarter"].unique()) <= {0, 1}
    assert df.loc[df["converted_next_quarter"] == 0, "spend_90d_inr"].isna().all()
    assert df.loc[df["converted_next_quarter"] == 1, "spend_90d_inr"].notna().all()


def test_spend_is_positive_for_converters(tiny_panel):
    _, df = tiny_panel
    spend = df.loc[df["converted_next_quarter"] == 1, "spend_90d_inr"]
    assert (spend > 0).all()


def test_no_post_conversion_prospect_rows(tiny_panel):
    """A converted brand shows 0s until its single conversion row, and
    never appears as a prospect again afterward."""
    _, df = tiny_panel
    converted_brands = df.loc[df["converted_next_quarter"] == 1, "brand_id"]
    for b, grp in df[df["brand_id"].isin(converted_brands)].groupby("brand_id"):
        labels = grp.sort_values("snapshot_quarter")["converted_next_quarter"].tolist()
        # exactly one 1, and it is the LAST observation for that brand
        assert labels.count(1) == 1, f"multiple conversions for {b}"
        assert labels[-1] == 1, f"prospect row appears after conversion for {b}"
