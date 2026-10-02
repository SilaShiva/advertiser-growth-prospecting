"""Shared fixtures: tiny synthetic panel so tests stay fast (<10s)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tiny_panel(cfg):
    """A small panel generated with the real generator (config
    overridden to more quarters so splits still work)."""
    import copy
    c = copy.deepcopy(cfg)
    c["generation"]["n_brands"] = 150
    c["generation"]["quarters"] = 10
    from src.ingestion.generate_data import generate_snapshots
    df = generate_snapshots(c)
    return c, df


@pytest.fixture(scope="session")
def tiny_matrix(tiny_panel):
    """(cfg, panel, stats, matrix) with stats fit on train quarters only."""
    c, panel = tiny_panel
    from src.features.build_features import fit_stats, build_features
    quarters = sorted(panel["snapshot_quarter"].unique())
    train = panel[panel["snapshot_quarter"].isin(quarters[:c["splits"]["train_quarters"]])]
    stats = fit_stats(train, c)
    matrix = build_features(panel, stats, c)
    return c, panel, stats, matrix
