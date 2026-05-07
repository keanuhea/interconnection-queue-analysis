"""Streamlit dashboard for the U.S. interconnection queue analysis.

Designed for an executive audience: leads with the problem, surfaces the
biggest signal first, and keeps the methodology visible (collapsed) for trust.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from datetime import date

from src.concentration_analysis import (
    POI_SENTINELS,
    concentration_summary,
    top_concentration,
)
from src.forward_sim import simulate
from src.load_data import COLUMN_MAP, QUEUE_SHEET, find_data_file, load_queued_up
from src.pjm_queue import list_snapshots, load_snapshot
from src.pjm_scoring import score_pjm_active
from src.state_machine import (
    ACTIVE_STATES,
    CANONICAL_TRANSITIONS,
    State,
    cohort_from_lbnl,
    fit_hazards,
)
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


@st.cache_data(show_spinner="Loading latest PJM snapshot...")
def _load_pjm():
    snapshots = list_snapshots()
    if not snapshots:
        return None, None
    latest = snapshots[-1]
    return load_snapshot(latest), date.fromisoformat(latest.stem)


@st.cache_data(show_spinner="Scoring PJM active queue...")
def _score_pjm(_pjm_df, _lbnl_df):
    return score_pjm_active(_pjm_df, _lbnl_df)


@st.cache_resource(show_spinner="Fitting transition hazards...")
def _fit_hazards(_df):
    return fit_hazards(_df)


@st.cache_data(show_spinner="Running 500 forward simulations...")
def _simulate(_df, horizon_years: int = 10, n_replicates: int = 500):
    table = _fit_hazards(_df)
    cohort = cohort_from_lbnl(_df, table.asof)
    result = simulate(cohort, table, horizon_years=horizon_years, n_replicates=n_replicates)
    return table, cohort, result


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

# ───── Live PJM Queue Tracker ─────────────────────────────────────────────────
pjm_df, snapshot_dt = _load_pjm()

if pjm_df is not None:
    st.header("Live tracker: PJM queue right now")
    st.caption(
        f"Snapshot taken **{snapshot_dt:%B %d, %Y}** directly from PJM's planning API. "
        "PJM operates the largest U.S. RTO (67M people, 13 states + DC) and is Tapestry's "
        "first deployment partner for HyperQ. Cycle 1 of PJM's reformed interconnection process "
        "received 811 new projects (220 GW) on April 28, 2026 — that data is in PJM's 91-day "
        "validation phase and not yet machine-readable. The numbers below cover the **transition cohort**: "
        "projects already in PJM's queue working through the legacy → reformed handoff."
    )

    pjm_active = pjm_df[pjm_df["Status"] == "Active"]
    pjm_inflight = pjm_df[pjm_df["Status"].isin(
        ["Active", "Engineering and Procurement", "Confirmed", "Suspended", "Under Construction"]
    )]
    pjm_active_gw = pjm_active["MW Capacity"].fillna(pjm_active["MW Energy"]).sum() / 1000

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Active in PJM queue", f"{len(pjm_active):,}")
    p2.metric("Active capacity", f"{pjm_active_gw:,.0f} GW")
    p3.metric(
        "In-flight (all live phases)",
        f"{len(pjm_inflight):,}",
        help="Active + Engineering & Procurement + Confirmed + Suspended + Under Construction.",
    )
    p4.metric("Snapshot date", snapshot_dt.strftime("%Y-%m-%d"))

    scored_pjm = _score_pjm(pjm_df, df)

    pcol1, pcol2 = st.columns([3, 2])

    with pcol1:
        st.subheader("Highest withdrawal-risk active projects")
        st.caption(
            "Each active project scored with the LBNL-trained gradient-boosting model "
            "(features: queue age, MW, resource type, RTO). Useful for spotting projects "
            "that historically resemble withdrawn ones."
        )
        risk_view = (
            scored_pjm.sort_values("p_withdraw", ascending=False)
            .head(25)[["queue_id", "project_name", "State", "Fuel",
                       "mw", "queue_age_years", "p_withdraw"]]
            .rename(columns={
                "queue_id": "Queue ID",
                "project_name": "Project",
                "Fuel": "Resource",
                "mw": "MW",
                "queue_age_years": "Age (yrs)",
                "p_withdraw": "P(withdraw)",
            })
        )
        st.dataframe(
            risk_view.style.format(
                {"MW": "{:,.0f}", "Age (yrs)": "{:.1f}", "P(withdraw)": "{:.0%}"}
            ),
            use_container_width=True,
            hide_index=True,
            height=380,
        )

    with pcol2:
        st.subheader("Active queue composition")
        fuel_counts = (
            pjm_active.groupby("Fuel")
            .agg(projects=("Project ID", "size"), total_mw=("MW Capacity", "sum"))
            .reset_index()
            .sort_values("projects", ascending=False)
        )
        fuel_counts["GW"] = (fuel_counts["total_mw"] / 1000).round(1)
        fig = px.bar(
            fuel_counts,
            x="Fuel",
            y="projects",
            text="projects",
            title="Active PJM projects by fuel type",
            labels={"Fuel": "", "projects": "Active projects"},
            height=380,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cycle 1 (reformed process)")
    cyc1, cyc2, cyc3, cyc4 = st.columns(4)
    cyc1.metric("Applications received", "811", help="As announced by PJM April 29, 2026.")
    cyc2.metric("Total nameplate capacity", "220 GW")
    cyc3.metric("Validation window", "Apr 28 – Jul 27 2026")
    cyc4.metric("Phase I begins", "~ Jul 28 2026")
    st.caption(
        "Cycle 1 composition (per PJM): 349 storage · 157 gas · 142 solar · 65 wind · "
        "45 solar+storage · 27 nuclear · 11 hydro · 15 other (incl. fusion). "
        "Per-project data isn't published yet — the tracker scaffold is ready to ingest it "
        "as soon as PJM exposes the new feed."
    )

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

# ───── Section 3: Forward simulation ──────────────────────────────────────────
table, cohort, sim = _simulate(df, horizon_years=10, n_replicates=500)
horizon_idx = len(sim.months) - 1
horizon_dt = sim.months[-1]

initial_gw = cohort["mw"].fillna(0).sum() / 1000
op_gw_p50 = float(sim.operational_gw_quantiles((0.5,)).iloc[-1, 0])
op_gw_p10 = float(sim.operational_gw_quantiles((0.1,)).iloc[-1, 0])
op_gw_p90 = float(sim.operational_gw_quantiles((0.9,)).iloc[-1, 0])

horizon_states = sim.state_at_horizon(horizon_idx)
expected_op = float(horizon_states.loc["operational", "mean"])
expected_wd = float(horizon_states.loc["withdrawn", "mean"])
expected_stuck = float(
    horizon_states.loc[[s.value for s in ACTIVE_STATES], "mean"].sum()
)
n_cohort = len(cohort)

st.header(
    f"If history repeats: only {expected_op / n_cohort:.0%} of today's queue reaches the grid by {horizon_dt.year}"
)
st.markdown(
    f"Starting from **{n_cohort:,} active LBNL projects ({initial_gw:,.0f} GW)** and rolling forward "
    "ten years using empirically-fit monthly transition hazards, the simulation runs **500 Monte Carlo "
    "replicates**. Each replicate samples a possible future for every project independently, given its "
    "current milestone state. The fan chart below shows the resulting distribution of operational GW "
    "over time — the spread is queue-progression uncertainty, not measurement noise."
)

s1, s2, s3, s4 = st.columns(4)
s1.metric(
    f"Projects operational by {horizon_dt.year}",
    f"{expected_op:,.0f}",
    help=f"Mean across 500 replicates. P10–P90: "
         f"{horizon_states.loc['operational', 'p10']:,.0f} – "
         f"{horizon_states.loc['operational', 'p90']:,.0f}.",
)
s2.metric(
    "Expected operational GW",
    f"{op_gw_p50:,.0f} GW",
    help=f"Median (P50). P10: {op_gw_p10:,.0f} GW · P90: {op_gw_p90:,.0f} GW.",
)
s3.metric(
    f"Projected withdrawals by {horizon_dt.year}",
    f"{expected_wd:,.0f}",
    help=f"Mean across replicates. P10–P90: "
         f"{horizon_states.loc['withdrawn', 'p10']:,.0f} – "
         f"{horizon_states.loc['withdrawn', 'p90']:,.0f}.",
)
s4.metric(
    "Still in queue at horizon",
    f"{expected_stuck:,.0f}",
    help="Projects that have neither reached commercial operation nor withdrawn after ten years.",
)

# Fan chart: operational GW over time
quantiles = sim.operational_gw_quantiles((0.1, 0.5, 0.9))
fan_df = quantiles.reset_index().rename(columns={"month": "date"})

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=fan_df["date"], y=fan_df["p90"], line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="P90",
    )
)
fig.add_trace(
    go.Scatter(
        x=fan_df["date"], y=fan_df["p10"], line=dict(width=0),
        fill="tonexty", fillcolor="rgba(99, 110, 250, 0.20)",
        name="P10–P90 envelope", hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=fan_df["date"], y=fan_df["p50"],
        line=dict(color="rgb(99, 110, 250)", width=3),
        name="Median (P50)",
        hovertemplate="%{x|%b %Y}: %{y:,.0f} GW operational<extra></extra>",
    )
)
fig.update_layout(
    title=f"Operational GW from today's active cohort, {horizon_dt.year - 10} → {horizon_dt.year}",
    xaxis_title="",
    yaxis_title="Operational GW (cumulative)",
    height=400,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# Stacked area: state composition over time
share_df = sim.state_share_mean().reset_index().melt(
    id_vars="month", var_name="state", value_name="projects"
)
state_order = ["submitted", "ia_signed", "operational", "withdrawn"]
share_df["state"] = pd.Categorical(share_df["state"], categories=state_order, ordered=True)
share_df = share_df.sort_values(["month", "state"])

fig = px.area(
    share_df,
    x="month",
    y="projects",
    color="state",
    title="Where the cohort goes: average project counts by state over time",
    labels={"month": "", "projects": "Projects (mean across replicates)", "state": "State"},
    category_orders={"state": state_order},
    color_discrete_map={
        "submitted": "#aaa",
        "ia_signed": "#7e9bff",
        "operational": "#3ec47e",
        "withdrawn": "#e85d75",
    },
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📐 How the simulation is fit (methodology)", expanded=False):
    st.markdown(
        "**Model.** Continuous-time Markov chain over four states "
        "(`submitted`, `ia_signed`, `operational`, `withdrawn`), discretized at monthly "
        "resolution. Each active project independently samples a transition each month "
        "from a categorical distribution. Hazards are piecewise-constant — the simplest "
        "defensible model given LBNL only records milestone *dates*, not per-month status."
    )
    st.markdown(
        f"**Calibration window.** All LBNL projects entering the queue 2010-01-01 through "
        f"{table.asof.date()}, exposure-weighted. Older entries are excluded because LBNL "
        "carries some pre-2003 sentinel records."
    )
    st.markdown("**Empirical monthly hazards** (per active state):")
    rows = []
    for tr in CANONICAL_TRANSITIONS:
        p_m = table.monthly_p[tr.from_state][tr.to_state]
        p_y = 1 - (1 - p_m) ** 12
        n = table.n_observed[tr.from_state][tr.to_state]
        rows.append({
            "From": tr.from_state.value,
            "To": tr.to_state.value,
            "Monthly P": f"{p_m:.4f}",
            "Annualized P": f"{p_y:.1%}",
            "n observed": f"{n:,}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "**Caveats.** (1) Hazards are pooled across RTO and resource type; the next "
        "iteration will fit per (state × RTO × resource). "
        "(2) Right-censored projects contribute to exposure but not to transition counts — "
        "the standard treatment, but it underweights the actual risk of older lingering projects. "
        "(3) The model assumes hazards are stationary; FERC Order 2023's reformed cluster "
        "process is *not* yet visible in the calibration data, so the baseline projects forward "
        "from pre-reform dynamics. The what-if layer (next) lets you explore scenarios where it isn't."
    )

st.divider()

# ───── Section 4: How we got here ─────────────────────────────────────────────
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
