"""Identify high-concentration interconnection points.

A POI (point of interconnection / substation) is "concentrated" when many
projects with significant total MW are stacked on it. We flag the top decile
by total queued MW as risk clusters.

Run as a script for a printed summary; import `concentration_summary` for
programmatic use (e.g., the dashboard).
"""

from __future__ import annotations

import pandas as pd

from src.load_data import load_queued_up


def concentration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate queue load per POI and flag top-decile concentration."""
    if "poi" not in df.columns:
        raise KeyError(
            "Expected a 'poi' column. Check load_data.CANONICAL_COLUMNS — "
            "your dataset's POI column may be named differently."
        )

    open_df = df[df["withdrawn"] == 0].copy()
    grouped = (
        open_df.groupby("poi", dropna=True)
        .agg(
            project_count=("poi", "size"),
            total_mw=("mw", "sum"),
            mean_age_years=("queue_age_years", "mean"),
            rtos=("rto", lambda s: ",".join(sorted(set(str(x) for x in s if pd.notna(x))))),
        )
        .reset_index()
    )
    grouped = grouped.sort_values("total_mw", ascending=False).reset_index(drop=True)

    if not grouped.empty:
        threshold = grouped["total_mw"].quantile(0.9)
        grouped["risk_cluster"] = grouped["total_mw"] >= threshold

    return grouped


def top_concentration(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return concentration_summary(df).head(n)


def upgrade_cost_corridors(df: pd.DataFrame, cost_col: str | None = None) -> pd.DataFrame:
    """Aggregate upgrade costs by RTO and POI when cost data is present.

    Queued Up does not always include upgrade cost columns; if none are found
    the function returns an empty frame with a message. Pass `cost_col`
    explicitly if your dataset has a non-standard column name.
    """
    candidate_cols = [
        cost_col,
        "upgrade_cost",
        "network_upgrade_cost",
        "estimated_upgrade_cost",
        "transmission_upgrade_cost",
    ]
    available = [c for c in candidate_cols if c and c in df.columns]
    if not available:
        print(
            "No upgrade cost column found. Pass `cost_col=` explicitly if your "
            "dataset has one (Queued Up only includes cost data for some RTOs)."
        )
        return pd.DataFrame()

    col = available[0]
    open_df = df[df["withdrawn"] == 0].copy()
    open_df[col] = pd.to_numeric(open_df[col], errors="coerce")
    grouped = (
        open_df.dropna(subset=[col])
        .groupby(["rto", "poi"], dropna=True)[col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "total_cost", "mean": "mean_cost"})
        .sort_values("total_cost", ascending=False)
    )
    return grouped


if __name__ == "__main__":
    df = load_queued_up()
    summary = concentration_summary(df)

    print(f"Unique POIs: {len(summary):,}")
    if "risk_cluster" in summary.columns:
        flagged = summary[summary["risk_cluster"]]
        print(f"Top-decile risk clusters: {len(flagged):,} POIs")
        print(f"  Total MW in risk clusters: {flagged['total_mw'].sum():,.0f}")
        print(f"  Share of all queued MW: "
              f"{flagged['total_mw'].sum() / summary['total_mw'].sum():.1%}")

    print("\nTop 20 highest-concentration POIs:")
    print(top_concentration(df, 20).to_string(index=False))

    costs = upgrade_cost_corridors(df)
    if not costs.empty:
        print("\nTop 20 corridors by total network upgrade cost:")
        print(costs.head(20).to_string(index=False))
