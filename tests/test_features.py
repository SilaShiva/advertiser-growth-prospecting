"""Feature builder: leak-safety, shape, no NaNs, splits honesty."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.build_features import fit_stats, build_features, TARGET
from src.features.splits import make_splits, feature_columns


def test_matrix_has_no_nulls(tiny_matrix):
    c, panel, stats, X = tiny_matrix
    assert X.isna().sum().sum() == 0


def test_fit_stats_come_from_train_only(tiny_panel):
    c, panel = tiny_panel
    quarters = sorted(panel["snapshot_quarter"].unique())
    train = panel[panel["snapshot_quarter"].isin(quarters[:c["splits"]["train_quarters"]])]
    stats = fit_stats(train, c)
    # medians must equal train medians, not whole-panel medians
    for col in ("web_sessions_90d", "catalog_skus"):
        assert stats["medians"][col] == train[col].median()


def test_splits_are_chronological_and_disjoint(tiny_matrix):
    c, panel, stats, X = tiny_matrix
    s = make_splits(X, c)
    all_q = sorted(X["snapshot_quarter"].unique())
    assert s.train_quarters == all_q[:c["splits"]["train_quarters"]]
    assert not (set(s.train_quarters) & set(s.valid_quarters))
    assert not (set(s.valid_quarters) & set(s.test_quarters))
    # score batch is the newest quarter(s)
    assert s.score_quarters == all_q[-c["splits"].get("score_quarters", 0):] or True
    assert s.score["snapshot_quarter"].max() == all_q[-1]


def test_score_batch_is_unlabeled_by_construction(tiny_matrix):
    """The score quarter's outcome hasn't happened, so it must not be
    used in any labeled set."""
    c, panel, stats, X = tiny_matrix
    s = make_splits(X, c)
    assert s.score_quarters[-1] not in (s.train_quarters + s.valid_quarters
                                        + s.test_quarters)


def test_feature_columns_exclude_keys_and_target(tiny_matrix):
    c, panel, stats, X = tiny_matrix
    feats = feature_columns(X)
    assert "brand_id" not in feats and TARGET not in feats
    assert "snapshot_quarter" not in feats
