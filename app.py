"""Streamlit dashboard for the U.S. interconnection queue analysis.

Designed for an executive audience: leads with the problem, surfaces the
biggest signal first, and keeps the methodology visible (collapsed) for trust.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.concentration_analysis import (
    POI_SENTINELS,
    concentration_summary,
    top_concentration,
)
from src.load_data import COLUMN_MAP, QUEUE_SHEET, find_data_file, load_queued_up
from src.withdrawal_model import score_open_queue, train

st.set_page_config(
    page_title="U.S. Interconnection Queue: where the grid is stuck",
    layout="wide",
)


@st.cache_data(show_spinner="Loading Queued Up dataset...")
def _load() -> pd.DataFrame:
    return load_queued_up()


@st.cache_resource(show_spinner="Training withdrawal model...")
def _train(df: pd.DataFrame):
    return train(df)


try:
    df = _load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

resolved = df[(df["withdrawn"] == 1) | (df["operational"] == 1)]
active = df[df["status"] == "active"]

withdrawal_rate_resolved = resolved["withdrawn"].mean() if len(resolved) else 0
completion_rate_resolved = resolved["operational"].mean() if len(resolved) else 0
total_active_gw = active["mw"].sum() / 1000
median_active_wait = active["queue_age_years"].median()

# ───── Headline ───────────────────────────────────────────────────────────────
st.title("The U.S. grid is stuck waiting in line")
st.markdown(
    f"**{len(active):,} projects** representing **{total_active_gw:,.0f} GW** of new "
    "generation capacity are currently waiting in U.S. interconnection queues — "
    "roughly **1.6× the entire installed capacity of the U.S. grid today**. "
    f"Of projects that have already resolved, only **{completion_rate_resolved:.0%}** "
    "ever reach commercial operation."
)
st.caption(
    f"Source: Berkeley Lab *Queued Up* 2025 edition (data through 2024-12-31). "
    f"{len(df):,} project records across 9 RTOs/regions. "
    "Built as a portfolio piece exploring the same data problem Tapestry (Alphabet) is solving for grid operators."
)

st.divider()

# ───── Hero KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Active queue",
    f"{len(active):,} projects",
    help="Projects currently with status = active (neither withdrawn nor operational).",
)
k2.metric("Capacity waiting", f"{total_active_gw:,.0f} GW")
k3.metric(
    "Historical completion rate",
    f"{completion_rate_resolved:.0%}",
    help="Among resolved projects (withdrawn or operational), share that reached commercial operation.",
)
k4.metric(
    "Median wait of active projects",
    f"{median_active_wait:.1f} years",
    help="Years between queue entry date and 2024-12-31.",
)

# ───── How the data is being read ─────────────────────────────────────────────
with st.expander("📂 How the data is being read (methodology)", expanded=False):
    src_file = find_data_file().name
    qmin = df["queue_date"].dropna().quantile(0.05).year
    qmax = df["queue_date"].dropna().max().year

    st.markdown(
        f"""
**Source file**: `{src_file}` (Berkeley Lab Queued Up 2025 edition)
**Sheet used**: `{QUEUE_SHEET}` — header read from row 2 (row 1 is a *RETURN TO CONTENTS* banner)
**Records loaded**: {len(df):,}
**Queue entry dates**: 5th-percentile {qmin} → max {qmax}
(A small number of pre-2003 entries appear to be sentinel/missing values; charts below filter to 2010+.)
"""
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Status breakdown** (raw `q_status` from the dataset)")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Projects"]
        status_counts["Share"] = (
            status_counts["Projects"] / len(df) * 100
        ).round(1).astype(str) + "%"
        st.dataframe(status_counts, use_container_width=True, hide_index=True)

        st.markdown("**Label derivation**")
        st.caption(
            "**withdrawn = 1** if `q_status` contains *withdraw* OR `wd_date` is set.  \n"
            "**operational = 1** if `q_status` matches *operating | in service | operational | commercial* "
            "OR `on_date` is set.  \n"
            f"**POI sentinels excluded** from concentration analysis: `{', '.join(sorted(POI_SENTINELS))}`."
        )

    with right:
        st.markdown("**Column mapping** (raw → canonical)")
        mapping_df = pd.DataFrame(
            list(COLUMN_MAP.items()),
            columns=["Raw column (LBNL)", "Canonical name (this dashboard)"],
        )
        st.dataframe(mapping_df, use_container_width=True, hide_index=True, height=400)

st.divider()

# ───── Section 1: Concentration ───────────────────────────────────────────────
full_summary = concentration_summary(df)
risk = full_summary[full_summary["risk_cluster"]]
share_top10 = risk["total_mw"].sum() / full_summary["total_mw"].sum() if not full_summary.empty else 0

st.header(f"Bottleneck: top 10% of substations carry {share_top10:.0%} of queued capacity")
st.markdown(
    "A **POI** (point of interconnection) is the substation where a new generation project "
    "plugs into the grid. When many projects target the same POI, they all wait on the same "
    "network-upgrade studies — and any one project's delay propagates to its cluster mates."
)

n = st.slider("Substations to display", 10, 50, 20, key="poi_n")
top = top_concentration(df, n)

fig = px.bar(
    top.iloc[::-1],
    x="total_mw",
    y="poi",
    color="rtos",
    orientation="h",
    title=f"Top {n} substations by total queued MW (excludes withdrawn projects)",
    labels={"total_mw": "Total queued MW", "poi": "Substation", "rtos": "RTO"},
    height=max(420, 28 * n),
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"**{len(risk):,} POIs** are flagged as risk clusters (top decile by total queued MW). "
    f"They represent **{share_top10:.0%}** of all non-withdrawn capacity in the dataset."
)

st.divider()

# ───── Section 2: Withdrawal ──────────────────────────────────────────────────
st.header(
    f"Most projects don't make it: {withdrawal_rate_resolved:.0%} of resolved requests are withdrawn"
)
st.markdown(
    "Of every 100 projects that finish their interconnection journey — either reaching commercial "
    f"operation or being formally withdrawn — only about **{completion_rate_resolved * 100:.0f}** "
    "actually get built. The model below quantifies which features most predict withdrawal so "
    "we can flag at-risk active projects."
)

clf, encoder, importances = _train(df)

mcol1, mcol2 = st.columns([2, 3])

def _prettify_feature(name: str) -> str:
    if name == "queue_age_years":
        return "Queue age (years)"
    if name == "mw":
        return "Capacity (MW)"
    if name.startswith("rto_"):
        return f"RTO: {name[len('rto_'):]}"
    if name.startswith("resource_type_"):
        return f"Resource: {name[len('resource_type_'):]}"
    return name.replace("_", " ").capitalize()


with mcol1:
    st.subheader("What drives withdrawal?")
    importances_df = importances.head(10).reset_index()
    importances_df.columns = ["Feature", "Importance"]
    importances_df["Feature"] = importances_df["Feature"].map(_prettify_feature)
    fig = px.bar(
        importances_df.iloc[::-1],
        x="Importance",
        y="Feature",
        orientation="h",
        labels={"Importance": "Relative importance", "Feature": ""},
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Gradient-boosted classifier on resolved projects only.  \n"
        "ROC-AUC: **0.81** · PR-AUC: **0.96** · base rate: **83% withdrawn**.  \n"
        "Caveat: queue age dominates partly because older entries have had more time to resolve."
    )

with mcol2:
    st.subheader("Which active projects are most at risk?")
    scored = score_open_queue(df, clf, encoder)
    if not scored.empty:
        view = (
            scored[
                ["project_name", "rto", "resource_type", "mw",
                 "queue_age_years", "p_withdraw"]
            ]
            .sort_values("p_withdraw", ascending=False)
            .head(50)
            .rename(
                columns={
                    "project_name": "Project",
                    "rto": "RTO",
                    "resource_type": "Resource",
                    "mw": "MW",
                    "queue_age_years": "Queue age (yrs)",
                    "p_withdraw": "P(withdraw)",
                }
            )
        )
        st.dataframe(
            view.style.format(
                {"MW": "{:,.0f}", "Queue age (yrs)": "{:.1f}", "P(withdraw)": "{:.0%}"}
            ),
            use_container_width=True,
            hide_index=True,
            height=400,
        )

# Mean P(withdraw) by RTO — useful exec view
rto_risk = (
    scored.groupby("rto")["p_withdraw"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"count": "Active projects", "mean": "Mean P(withdraw)"})
    .sort_values("Mean P(withdraw)", ascending=False)
)
fig = px.bar(
    rto_risk,
    x="rto",
    y="Mean P(withdraw)",
    title="Predicted withdrawal probability for the active queue, by RTO",
    labels={"rto": "RTO", "Mean P(withdraw)": "Mean P(withdraw)"},
    text=rto_risk["Mean P(withdraw)"].map("{:.0%}".format),
)
fig.update_traces(textposition="outside")
fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ───── Section 3: How we got here ─────────────────────────────────────────────
st.header("How we got here: queue growth has accelerated since 2018")

if "queue_date" in df.columns and "rto" in df.columns:
    plot_df = df.dropna(subset=["queue_date", "rto"]).copy()
    plot_df["year"] = plot_df["queue_date"].dt.year
    plot_df = plot_df[plot_df["year"].between(2010, 2024)]

    by_year = plot_df.groupby(["year", "rto"]).size().reset_index(name="projects")
    fig = px.area(
        by_year,
        x="year",
        y="projects",
        color="rto",
        title="New interconnection requests per year, by RTO",
        labels={"year": "Queue entry year", "projects": "Projects entering queue", "rto": "RTO"},
    )
    st.plotly_chart(fig, use_container_width=True)

if "resource_type" in df.columns and "queue_date" in df.columns and "mw" in df.columns:
    rt_df = df.dropna(subset=["queue_date", "resource_type", "mw"]).copy()
    rt_df["year"] = rt_df["queue_date"].dt.year
    rt_df = rt_df[rt_df["year"].between(2010, 2024)]
    rt_agg = (
        rt_df.groupby(["year", "resource_type"])["mw"]
        .sum()
        .reset_index()
    )
    rt_agg["GW"] = rt_agg["mw"] / 1000
    fig = px.area(
        rt_agg,
        x="year",
        y="GW",
        color="resource_type",
        title="GW entering queue per year, by resource type",
        labels={"year": "Queue entry year", "GW": "GW entering queue", "resource_type": "Resource"},
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Built with pandas + scikit-learn + plotly + streamlit. "
    "Source code: github.com/keanuhea/interconnection-queue-analysis"
)
