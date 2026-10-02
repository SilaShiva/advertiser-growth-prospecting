"""Time-based splits and label assembly.

Splits are by snapshot quarter, never random: training sees only past
quarters, validation the next ones, and the final quarter plays the role
of a fresh production batch. Within training, cross-validation is
GroupKFold on quarter so no quarter's rows leak across folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_features import TARGET


@dataclass
class SplitSets:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    score: pd.DataFrame          # newest quarter: unlabeled production batch
    train_quarters: list[str]
    valid_quarters: list[str]
    test_quarters: list[str]
    score_quarters: list[str]


def make_splits(matrix: pd.DataFrame, cfg: dict) -> SplitSets:
    """Split the model matrix by snapshot quarter (chronological).

    The final `score_quarters` snapshot(s) are returned separately as the
    unlabeled production batch — their outcome quarter hasn't closed, so
    they carry no label and are never used for evaluation.
    """
    scfg = cfg["splits"]
    quarters = sorted(matrix["snapshot_quarter"].unique())
    n_train, n_valid, n_test = (scfg["train_quarters"], scfg["valid_quarters"],
                                scfg["test_quarters"])
    n_score = scfg.get("score_quarters", 0)
    if len(quarters) < n_train + n_valid + n_test + n_score:
        raise ValueError(
            f"need >= {n_train + n_valid + n_test + n_score} quarters, "
            f"found {len(quarters)}")
    train_q = quarters[:n_train]
    valid_q = quarters[n_train:n_train + n_valid]
    test_q = quarters[n_train + n_valid:n_train + n_valid + n_test]
    score_q = quarters[n_train + n_valid + n_test:]

    return SplitSets(
        train=matrix[matrix["snapshot_quarter"].isin(train_q)].copy(),
        valid=matrix[matrix["snapshot_quarter"].isin(valid_q)].copy(),
        test=matrix[matrix["snapshot_quarter"].isin(test_q)].copy(),
        score=matrix[matrix["snapshot_quarter"].isin(score_q)].copy(),
        train_quarters=train_q, valid_quarters=valid_q,
        test_quarters=test_q, score_quarters=score_q,
    )


def feature_columns(matrix: pd.DataFrame) -> list[str]:
    """All model inputs = every column except keys and the target."""
    return [c for c in matrix.columns
            if c not in ("brand_id", "snapshot_quarter", TARGET)]


def spend_labels(raw_panel: pd.DataFrame) -> pd.DataFrame:
    """First-90-day spend for converted rows, keyed on (brand, quarter).

    The spend model trains only on converted rows: predicting spend for
    non-converters is undefined, and at inference the two models compose
    as expected_spend = P(convert) * E[spend | convert].
    """
    sp = raw_panel[["brand_id", "snapshot_quarter", "spend_90d_inr"]].dropna()
    sp = sp.rename(columns={"spend_90d_inr": "spend_90d"})
    return sp


def group_folds(snapshot_quarters: pd.Series, n_folds: int) -> np.ndarray:
    """Fold ids by quarter group (sklearn GroupKFold-compatible order)."""
    uniq = sorted(snapshot_quarters.unique())
    fold_of = {q: i % n_folds for i, q in enumerate(uniq)}
    return snapshot_quarters.map(fold_of).to_numpy()
