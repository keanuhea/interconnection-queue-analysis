"""Score live PJM queue projects with the LBNL-trained withdrawal model.

PJM's queue export uses different column names and `Fuel` values than LBNL's
Queued Up dataset. This module bridges the two: it picks the active PJM rows,
maps them onto the LBNL canonical schema the model was trained against, then
returns `p_withdraw` per project.
"""

from __future__ import annotations

import pandas as pd

from src.load_data import load_queued_up
from src.withdrawal_model import encode_features, train

PJM_FUEL_TO_LBNL_RESOURCE = {
    "Solar": "Solar",
    "Storage": "Battery",
    "Solar; Storage": "Solar+Battery",
    "Wind": "Wind",
    "Offshore Wind": "Wind",
    "Natural Gas": "Gas",
    "Natural Gas; Other": "Gas",
    "Natural Gas; Oil": "Gas",
    "Nuclear": "Nuclear",
    "Other": "Other",
}


def _bridge_pjm_to_lbnl(
    pjm_df: pd.DataFrame, ref_date: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Project PJM active-queue rows onto the LBNL canonical schema."""
    df = pjm_df[pjm_df["Status"] == "Active"].copy()

    df["mw"] = df["MW Capacity"].fillna(df["MW Energy"]).fillna(df["MW In Service"])
    df["mw"] = pd.to_numeric(df["mw"], errors="coerce")

    submitted = pd.to_datetime(df["Submitted Date"], errors="coerce")
    ref = ref_date or pd.Timestamp.today().normalize()
    df["queue_age_years"] = ((ref - submitted).dt.days / 365.25).round(2).clip(lower=0)

    df["rto"] = "PJM"
    df["resource_type"] = df["Fuel"].map(PJM_FUEL_TO_LBNL_RESOURCE).fillna("Other")

    df["project_name"] = df["Name"]
    df["queue_id"] = df["Project ID"]
    df["transitioned"] = df["Project ID"].astype(str).str.contains("moved to", na=False)

    return df


def score_pjm_active(pjm_df: pd.DataFrame, lbnl_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train the LBNL model and score every active PJM project's P(withdraw).

    Returns a DataFrame with the bridged columns plus `p_withdraw`.
    Drops rows with missing MW or queue age (the model can't score them).
    """
    if lbnl_df is None:
        lbnl_df = load_queued_up()

    clf, encoder, _ = train(lbnl_df)

    bridged = _bridge_pjm_to_lbnl(pjm_df)
    bridged = bridged.dropna(subset=["mw", "queue_age_years"])

    X, _ = encode_features(bridged, encoder=encoder)
    bridged["p_withdraw"] = clf.predict_proba(X)[:, 1]

    return bridged


def summarize(scored: pd.DataFrame) -> None:
    print(f"Scored {len(scored):,} active PJM projects")
    print(f"Mean P(withdraw): {scored['p_withdraw'].mean():.1%}")
    print(f"Median P(withdraw): {scored['p_withdraw'].median():.1%}")
    print()
    print("Mean P(withdraw) by resource:")
    print(
        scored.groupby("resource_type")["p_withdraw"]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .round(3)
        .to_string()
    )
    print()
    print("Top 10 highest-risk active projects:")
    cols = ["queue_id", "project_name", "State", "resource_type", "mw",
            "queue_age_years", "p_withdraw"]
    cols = [c for c in cols if c in scored.columns]
    print(
        scored.nlargest(10, "p_withdraw")[cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    from src.pjm_queue import load_snapshot

    pjm = load_snapshot()
    scored = score_pjm_active(pjm)
    summarize(scored)
