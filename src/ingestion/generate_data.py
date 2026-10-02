"""Synthetic data generator: quarterly prospect-brand snapshots.

Emulates the data shape an ads-sales team would have about prospective
brands in a quick-commerce marketplace:

- one row per (brand, snapshot_quarter) until the brand converts;
- ~45 columns spanning web/app engagement, catalog health, category
  economics, competitive pressure and first-contact context;
- a latent conversion process (logistic in a weighted feature mix +
  noise) decides whether the brand onboards before the next snapshot;
- converted brands get a gamma-distributed first-90-day spend whose mean
  scales with their latent score.

Deterministic: fixed seed, no network, no external files. Running the
generator twice produces byte-identical CSVs (verified in tests), so
nothing sensitive ever needs to be committed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.loader import load_config

COLUMNS = [
    # identifiers / time
    "brand_id", "brand_name", "snapshot_date", "snapshot_quarter",
    "category", "region", "first_contact_date", "days_since_contact",
    # engagement
    "web_sessions_90d", "maus_est", "sessions_per_mau", "app_installs_est",
    "pdp_views_90d", "wishlist_adds_90d", "organic_search_rank_avg",
    # catalog health
    "catalog_skus", "in_stock_rate", "sku_velocity_index", "content_score",
    "freshness_days", "return_rate",
    # category economics
    "category_gmv_index", "category_growth_pct", "brand_gmv_share_pct",
    "avg_order_value", "discount_depth_pct",
    # competitive / pressure
    "competitor_spend_inr", "n_competitors_advertising",
    "share_of_voice_gap",
    # readiness / maturity
    "historical_ad_spend_inr", "digital_maturity", "agency_flag",
    "marketing_team_size", "has_d2c_store", "marketplace_rating",
    # contact context
    "contact_channel", "outreach_attempts", "demo_given", "last_action",
    # labels (present only for post-hoc evaluation on historic snapshots)
    "converted_next_quarter", "spend_90d_inr",
]

CATEGORY_GMV_BASE = {
    "Packaged Foods": 1.00, "Beverages": 0.92, "Personal Care": 1.10,
    "Household Care": 0.80, "Baby Care": 0.70, "Pet Supplies": 0.45,
    "Otc & Wellness": 0.88, "Snacks & Confectionery": 1.05,
}

LAST_ACTIONS = ["viewed_pricing", "requested_callback", "demo_scheduled",
                "uploaded_catalog", "ghosted", "negotiating"]
CONTACT_CHANNELS = ["inbound_form", "outbound_call", "partner_referral",
                    "field_sales", "email_campaign"]

BRAND_PREFIXES = ["Novara", "Brightleaf", "Kindroot", "Sunbeam", "Vitalis",
                  "Harvesta", "Purely", "Urbanbox", "Marigold", "Cobalt",
                  "Willowmere", "Terrafine", "Lumen", "Amberly", "Quillvale"]


def _brand_name(rng: np.random.Generator, idx: int) -> str:
    return f"{BRAND_PREFIXES[idx % len(BRAND_PREFIXES)]} {chr(65 + idx % 26)}{idx}"


def generate_snapshots(cfg: dict | None = None) -> pd.DataFrame:
    """Build the full (brand x quarter) panel with labels."""
    cfg = cfg or load_config()
    gen = cfg["generation"]
    rng = np.random.default_rng(cfg["project"]["random_seed"])
    n_brands = int(gen["n_brands"])
    quarters = int(gen["quarters"])

    cat_share = gen["category_share_top3"]
    categories = gen["categories"]
    shares = list(cat_share) + [(1 - sum(cat_share)) / (len(categories) - 3)] * (len(categories) - 3)

    rows: list[dict] = []
    for b in range(n_brands):
        brand_id = f"B{b:04d}"
        category = rng.choice(categories, p=shares)
        region = rng.choice(["north", "south", "west", "east"])
        first_contact = pd.Timestamp(gen["cohort_start"]) + pd.Timedelta(
            days=int(rng.integers(0, 60)))
        # Per-brand latent traits (stable across quarters) + fast-moving noise
        maturity = rng.beta(2.0, 4.0)                      # 0..1 digital maturity
        cat_gmv_base = CATEGORY_GMV_BASE[category]
        scale = rng.lognormal(mean=0.0, sigma=0.6)         # brand size scale
        converted = False
        onboard_lag = int(rng.integers(*gen["onboard_lag_days"]))

        for q in range(quarters):
            snap = pd.Timestamp(gen["cohort_start"]) + pd.DateOffset(months=3 * q)
            days_since = (snap - first_contact).days
            if days_since < 0:
                continue  # brand not yet contacted
            t = rng.normal(0, 0.15, size=1)[0]             # slow drift

            maus = int(max(50, scale * rng.lognormal(11.0, 0.5)))
            sessions = int(maus * rng.uniform(1.5, 6.0))
            skus = int(max(3, scale * rng.lognormal(4.2, 0.7)))
            hist_spend = float(max(0.0, scale * rng.lognormal(11.5, 0.9) * maturity))
            comp_spend = float(cat_gmv_base * scale * rng.lognormal(11.8, 0.8))

            row = {
                "brand_id": brand_id,
                "brand_name": _brand_name(rng, b),
                "snapshot_date": snap,
                "snapshot_quarter": f"{snap.year}Q{(snap.month - 1) // 3 + 1}",
                "category": category,
                "region": region,
                "first_contact_date": first_contact,
                "days_since_contact": days_since,
                "web_sessions_90d": sessions,
                "maus_est": maus,
                "sessions_per_mau": sessions / maus,
                "app_installs_est": int(maus * rng.uniform(0.4, 1.1)),
                "pdp_views_90d": int(sessions * rng.uniform(0.8, 2.4)),
                "wishlist_adds_90d": int(sessions * rng.uniform(0.02, 0.12)),
                "organic_search_rank_avg": float(rng.uniform(3.0, 40.0)),
                "catalog_skus": skus,
                "in_stock_rate": float(np.clip(rng.beta(8, 2), 0.3, 1.0)),
                "sku_velocity_index": float(skus / maus * 1000 * rng.uniform(0.5, 1.8)),
                "content_score": float(np.clip(0.3 + 0.6 * maturity + t, 0, 1)),
                "freshness_days": int(rng.exponential(20) + 1),
                "return_rate": float(np.clip(rng.beta(2, 30), 0, 0.3)),
                "category_gmv_index": float(cat_gmv_base * (1 + t * 0.1)),
                "category_growth_pct": float(rng.normal(9.0, 4.0)),
                "brand_gmv_share_pct": float(np.clip(rng.beta(1.5, 40) * 100, 0.01, 15)),
                "avg_order_value": float(rng.gamma(5.0, 45.0)),
                "discount_depth_pct": float(np.clip(rng.beta(3, 9) * 100, 0, 60)),
                "competitor_spend_inr": comp_spend,
                "n_competitors_advertising": int(rng.poisson(6 * cat_gmv_base) + 1),
                "share_of_voice_gap": float(np.clip(rng.normal(0.35, 0.2), -0.5, 1.0)),
                "historical_ad_spend_inr": hist_spend,
                "digital_maturity": float(np.clip(maturity + t, 0, 1)),
                # Readiness signals track maturity so the latent propensity
                # is genuinely learnable from observables (not independent noise).
                "agency_flag": int(rng.random() < 0.15 + 0.55 * maturity),
                "marketing_team_size": int(rng.poisson(1 + 6 * maturity)),
                "has_d2c_store": int(rng.random() < 0.10 + 0.55 * maturity),
                "marketplace_rating": float(np.clip(rng.normal(4.1, 0.4), 1.0, 5.0)),
                "contact_channel": rng.choice(CONTACT_CHANNELS),
                "outreach_attempts": int(rng.poisson(2.5) + 1),
                "demo_given": int(rng.random() < 0.10 + 0.55 * maturity),
                # ghosted probability falls with maturity
                "last_action": rng.choice(
                    LAST_ACTIONS,
                    p=[0.12, 0.14, 0.12, 0.14,
                       0.48 - 0.30 * maturity,      # ghosted
                       1.0 - (0.12 + 0.14 + 0.12 + 0.14 + 0.48 - 0.30 * maturity)]),
            }

            if not converted:
                # Quarterly conversion HAZARD driven by per-quarter observables
                # (demo, last action) plus stable brand traits. Negative
                # intercept keeps ~half the brands from ever converting inside
                # the window, so the label is a real decision, not timing luck.
                engaged_action = row["last_action"] in (
                    "requested_callback", "demo_scheduled", "negotiating",
                    "uploaded_catalog")
                hazard = (
                    -2.10                                        # base reluctance
                    + 1.8 * (maturity - 0.33) * 3.0              # brand trait
                    + 0.9 * row["demo_given"]                    # per-quarter signal
                    + 0.7 * float(engaged_action)                # per-quarter signal
                    + 0.5 * (row["has_d2c_store"])
                    + 0.6 * np.log1p(hist_spend) / 12.0
                    + 0.4 * np.log1p(comp_spend) / 14.0
                    + 0.3 * (min(days_since, 180) / 180.0)
                    - 0.4 * (row["last_action"] == "ghosted")
                )
                p_convert = 1 / (1 + np.exp(-(hazard / 0.55 + rng.normal(0, gen["conversion_noise_sd"]))))
                if rng.random() < p_convert and q < quarters - 1:
                    converted = True
                    onboard_date = snap + pd.DateOffset(days=onboard_lag)
                    spend_mean = 90_000 + 550_000 * (hazard + 1.8)
                    row["converted_next_quarter"] = 1
                    row["spend_90d_inr"] = float(max(
                        gen["min_spend_inr"],
                        rng.gamma(gen["spend_shape"],
                                  max(spend_mean, gen["min_spend_inr"]) / gen["spend_scale_divisor"])))
                    row["_onboard_date"] = onboard_date
                else:
                    row["converted_next_quarter"] = 0
                    row["spend_90d_inr"] = np.nan
            else:
                # Already converted in an earlier quarter: this snapshot is a
                # repeat row of an advertiser; exclude from prospect labels.
                row["converted_next_quarter"] = -1
                row["spend_90d_inr"] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    # Post-conversion repeat rows are dropped (a converted brand is no longer
    # a "prospect"); the -1 sentinel never reaches the pipeline.
    df = df[df["converted_next_quarter"] != -1].reset_index(drop=True)

    # Inject realistic missingness in two low-value columns.
    for col in gen["missing_seed_cols"]:
        if col not in df.columns:
            raise KeyError(f"missing_seed_cols references unknown column: {col}")
        mask = rng.random(len(df)) < gen["missing_rate"]
        df.loc[mask, col] = np.nan
    return df


def write_snapshot_csv(df: pd.DataFrame, out_dir, fmt_str: str = "%.6g") -> list:
    """Write one CSV per snapshot quarter; return written paths."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for q, grp in df.groupby("snapshot_quarter"):
        path = out_dir / f"prospects_{q}.csv"
        grp.drop(columns=["_onboard_date"], errors="ignore").to_csv(
            path, index=False, float_format=fmt_str)
        written.append(path)
    return written
