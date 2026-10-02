"""Data-quality gates: FAIL stops the pipeline, WARN only logs.

Each gate is a named check evaluated against the raw panel. Results are
collected into a report (JSON + markdown) under reports/data_quality/
for audit; the orchestrator refuses to continue on any FAIL.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class GateResult:
    name: str
    severity: str          # FAIL | WARN
    passed: bool
    detail: str


@dataclass
class GateReport:
    results: list[GateResult] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any((not r.passed) and r.severity == "FAIL" for r in self.results)

    @property
    def warnings(self) -> list[str]:
        return [f"{r.name}: {r.detail}" for r in self.results
                if (not r.passed) and r.severity == "WARN"]

    def to_dict(self) -> dict:
        return {"blocked": bool(self.blocked),
                "results": [{**r.__dict__, "passed": bool(r.passed)}
                            for r in self.results]}


def _gate_row_count_min(df: pd.DataFrame, value: int) -> tuple[bool, str]:
    ok = len(df) >= value
    return ok, f"{len(df):,} rows (min {value:,})"


def _gate_id_uniqueness(df: pd.DataFrame) -> tuple[bool, str]:
    dupes = df.duplicated(subset=["brand_id", "snapshot_quarter"]).sum()
    return dupes == 0, f"{int(dupes)} duplicate (brand, quarter) keys"


def _gate_label_parse_rate(df: pd.DataFrame, value: float) -> tuple[bool, str]:
    rate = float(df["converted_next_quarter"].isin([0, 1]).mean())
    return rate >= value, f"{rate:.4f} parseable labels (min {value})"


def _gate_label_values(df: pd.DataFrame) -> tuple[bool, str]:
    bad = set(df["converted_next_quarter"].dropna().unique()) - {0, 1}
    return len(bad) == 0, f"unexpected label values: {sorted(bad)}" if bad else "labels in {0,1}"


def _gate_null_share_numeric(df: pd.DataFrame, value: float) -> tuple[bool, str]:
    # Label columns are excluded: spend is NaN by design for non-converters.
    label_cols = {"converted_next_quarter", "spend_90d_inr"}
    numeric = df.select_dtypes(include="number").drop(columns=label_cols, errors="ignore")
    shares = numeric.isna().mean()
    worst = shares.idxmax()
    worst_share = float(shares.max())
    ok = worst_share <= value
    return ok, (f"worst null share {worst_share:.3f} in '{worst}' (max {value})")


def _gate_category_cardinality(df: pd.DataFrame, value: int) -> tuple[bool, str]:
    n = df["category"].nunique()
    return n == value, f"{n} categories (expected {value})"


def _gate_spend_nonnegative(df: pd.DataFrame) -> tuple[bool, str]:
    spend = df["spend_90d_inr"].dropna()
    bad = int((spend < 0).sum())
    return bad == 0, f"{bad} negative spend rows"


def _gate_duplicate_rows(df: pd.DataFrame, value: int) -> tuple[bool, str]:
    n = int(df.duplicated().sum())
    return n <= value, f"{n} exact duplicate rows (tolerated {value})"


_GATES = {
    "row_count_min":        lambda df, v: _gate_row_count_min(df, v),
    "id_uniqueness":        lambda df, _: _gate_id_uniqueness(df),
    "label_parse_rate":     lambda df, v: _gate_label_parse_rate(df, v),
    "label_values":         lambda df, _: _gate_label_values(df),
    "null_share_numeric":   lambda df, v: _gate_null_share_numeric(df, v),
    "category_cardinality": lambda df, v: _gate_category_cardinality(df, v),
    "spend_nonnegative":    lambda df, _: _gate_spend_nonnegative(df),
    "duplicate_rows":       lambda df, v: _gate_duplicate_rows(df, v),
}


def run_gates(df: pd.DataFrame, gate_specs: list[dict]) -> GateReport:
    """Evaluate configured gates in order; unknown gate names are FAIL."""
    report = GateReport()
    for spec in gate_specs:
        name, severity = spec["gate"], spec.get("severity", "WARN")
        fn = _GATES.get(name)
        if fn is None:
            report.results.append(GateResult(name, "FAIL", False,
                                             f"unknown gate '{name}' in config"))
            continue
        passed, detail = fn(df, spec.get("value"))
        report.results.append(GateResult(name, severity, passed, detail))
    return report


def save_report(report: GateReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "gate_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    lines = ["# Data-Quality Gate Report", "",
             f"**Blocked:** {'YES - pipeline stopped' if report.blocked else 'no'}", "",
             "| Gate | Severity | Passed | Detail |", "|---|---|---|---|"]
    for r in report.results:
        lines.append(f"| {r.name} | {r.severity} | "
                     f"{'PASS' if r.passed else 'FAIL'} | {r.detail} |")
    (out_dir / "gate_report.md").write_text("\n".join(lines), encoding="utf-8")
    return path
