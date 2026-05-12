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


@st.cache_data(show_spinner="Running scenario simulation...")
def _scenario(
    _df,
    study_speed: float,
    withdrawal_strict: float,
    build_speed: float,
    horizon_years: int = 10,
    n_replicates: int = 250,
):
    """Run the multi-state sim with operator-side lever multipliers applied.

    Three levers map onto four transitions (withdrawal lever scales both
    `→ withdrawn` paths). Returns a SimResult keyed by the slider values so
    the cache reuses identical lever combinations across reruns.
    """
    table = _fit_hazards(_df)
    cohort = cohort_from_lbnl(_df, table.asof)
    multipliers = {
        (State.SUBMITTED, State.IA_SIGNED): study_speed,
        (State.SUBMITTED, State.WITHDRAWN): withdrawal_strict,
        (State.IA_SIGNED, State.OPERATIONAL): build_speed,
        (State.IA_SIGNED, State.WITHDRAWN): withdrawal_strict,
    }
    return simulate(
        cohort, table,
        horizon_years=horizon_years,
        n_replicates=n_replicates,
        scenario_multipliers=multipliers,
    )


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
        named = scored[scored["project_name"].notna()].copy()
        named = named[named["queue_age_years"] >= 0]
        view = (
            named[
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
    "current milestone state. **Pull the levers below** to see how operator-side policy changes shift the cohort."
)

# ── Operator-side lever panel ────────────────────────────────────────────────
def _hazard_readout(table, study, strict, build):
    """Translate monthly hazards + lever multipliers into plain-English summary stats.

    Returns mean time-in-state and exit-share metrics so the slider captions can
    show 'baseline 4.2 yrs → scenario 2.8 yrs' alongside the multiplier itself.
    """
    h_ia = table.monthly_p[State.SUBMITTED][State.IA_SIGNED] * study
    h_wd_s = table.monthly_p[State.SUBMITTED][State.WITHDRAWN] * strict
    h_op = table.monthly_p[State.IA_SIGNED][State.OPERATIONAL] * build
    h_wd_i = table.monthly_p[State.IA_SIGNED][State.WITHDRAWN] * strict

    sub_total = h_ia + h_wd_s
    ia_total = h_op + h_wd_i

    sub_years = (1 / sub_total) / 12 if sub_total > 0 else float("inf")
    ia_years = (1 / ia_total) / 12 if ia_total > 0 else float("inf")
    p_advance_sub = h_ia / sub_total if sub_total > 0 else 0
    p_advance_ia = h_op / ia_total if ia_total > 0 else 0
    overall_op_pct = p_advance_sub * p_advance_ia * 100
    overall_wd_pct = 100 - overall_op_pct

    return {
        "sub_years": sub_years,
        "ia_years": ia_years,
        "overall_op_pct": overall_op_pct,
        "overall_wd_pct": overall_wd_pct,
    }


def _apply_preset(study, strict, build):
    st.session_state.study_speed = study
    st.session_state.withdrawal_strict = strict
    st.session_state.build_speed = build


for key, default in (("study_speed", 1.0), ("withdrawal_strict", 1.0), ("build_speed", 1.0)):
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("##### Try a named scenario, or drag the levers yourself")
p1, p2, p3, p4, p5 = st.columns(5)
if p1.button("Status quo", use_container_width=True,
             help="All levers at 1.0× — extrapolates the last decade forward."):
    _apply_preset(1.0, 1.0, 1.0)
    st.rerun()
if p2.button("Reform delivers", use_container_width=True,
             help="FERC Order 2023's cluster-study reform works: faster approvals, "
                  "stricter financial milestones cut speculative projects."):
    _apply_preset(1.5, 1.3, 1.0)
    st.rerun()
if p3.button("Construction crunch", use_container_width=True,
             help="Supply chain, transformer shortage, and labor squeeze slow projects "
                  "downstream of approval; developers hold positions instead of dropping out."):
    _apply_preset(1.0, 0.8, 0.6)
    st.rerun()
if p4.button("Reform stalls", use_container_width=True,
             help="Order 2023 doesn't deliver: studies still slow, withdrawal incentives "
                  "weak, construction roughly flat."):
    _apply_preset(0.8, 0.9, 1.0)
    st.rerun()
if p5.button("All systems go", use_container_width=True,
             help="Best-case 2030: faster studies, decisive withdrawals, "
                  "supply chain healed."):
    _apply_preset(1.6, 1.3, 1.4)
    st.rerun()

st.markdown(" ")

# Compute baseline + scenario readouts so each slider can show its real-world effect
base_read = _hazard_readout(table, 1.0, 1.0, 1.0)
sc_read = _hazard_readout(
    table,
    st.session_state.study_speed,
    st.session_state.withdrawal_strict,
    st.session_state.build_speed,
)

lv1, lv2, lv3, lv_reset = st.columns([3, 3, 3, 1])

with lv1:
    study_speed = st.slider(
        "How fast can RTOs approve a new project?",
        0.5, 2.0, key="study_speed", step=0.1,
        help="The cluster-study process — feasibility, system impact, facilities — "
             "is the regulatory bottleneck most reformers target. 1.5× is roughly "
             "what FERC Order 2023 was designed to achieve.",
    )
    st.markdown(
        f"<small>Avg time from queue entry to approval: "
        f"<b>{base_read['sub_years']:.1f} yrs</b> today → "
        f"<b style='color:#3ec47e'>{sc_read['sub_years']:.1f} yrs</b> scenario</small>",
        unsafe_allow_html=True,
    )

with lv2:
    withdrawal_strict = st.slider(
        "How aggressively are speculative projects culled?",
        0.5, 2.0, key="withdrawal_strict", step=0.1,
        help="Stricter financial milestones (deposits, ready-by deadlines) force "
             "speculative projects to drop out faster. 1.5× simulates the queue-reform "
             "intent: more withdrawals near-term, cleaner queue long-term. 0.7× is the "
             "opposite — projects park indefinitely.",
    )
    st.markdown(
        f"<small>Share of new projects that ever reach the grid: "
        f"<b>{base_read['overall_op_pct']:.0f}%</b> today → "
        f"<b style='color:#3ec47e'>{sc_read['overall_op_pct']:.0f}%</b> scenario</small>",
        unsafe_allow_html=True,
    )

with lv3:
    build_speed = st.slider(
        "How fast does construction actually finish?",
        0.5, 2.0, key="build_speed", step=0.1,
        help="Once a project has an interconnection agreement, can it actually get "
             "built? Captures transformer lead times, IRA tax-credit certainty, labor, "
             "and local permitting. 0.6× simulates the 2022–24 transformer shortage; "
             "1.4× simulates a healed supply chain.",
    )
    st.markdown(
        f"<small>Avg time from approval to operating: "
        f"<b>{base_read['ia_years']:.1f} yrs</b> today → "
        f"<b style='color:#3ec47e'>{sc_read['ia_years']:.1f} yrs</b> scenario</small>",
        unsafe_allow_html=True,
    )

with lv_reset:
    st.markdown("&nbsp;")
    if st.button("↺ Reset", help="Return all levers to baseline (1.0×)."):
        _apply_preset(1.0, 1.0, 1.0)
        st.rerun()

is_scenario = not (study_speed == 1.0 and withdrawal_strict == 1.0 and build_speed == 1.0)
scenario_sim = (
    _scenario(df, study_speed, withdrawal_strict, build_speed) if is_scenario else None
)

# Scenario aggregates for delta display
if scenario_sim is not None:
    sc_horizon = scenario_sim.state_at_horizon(len(scenario_sim.months) - 1)
    sc_op = float(sc_horizon.loc["operational", "mean"])
    sc_wd = float(sc_horizon.loc["withdrawn", "mean"])
    sc_stuck = float(sc_horizon.loc[[s.value for s in ACTIVE_STATES], "mean"].sum())
    sc_gw = float(scenario_sim.operational_gw_quantiles((0.5,)).iloc[-1, 0])

    delta_op = f"{sc_op - expected_op:+,.0f}"
    delta_gw = f"{sc_gw - op_gw_p50:+,.0f} GW"
    delta_wd = f"{sc_wd - expected_wd:+,.0f}"
    delta_stuck = f"{sc_stuck - expected_stuck:+,.0f}"
else:
    sc_op = sc_wd = sc_stuck = sc_gw = None
    delta_op = delta_gw = delta_wd = delta_stuck = None

s1, s2, s3, s4 = st.columns(4)
s1.metric(
    f"Projects operational by {horizon_dt.year}",
    f"{sc_op if is_scenario else expected_op:,.0f}",
    delta=delta_op,
    help=f"Baseline mean: {expected_op:,.0f} "
         f"(P10–P90 {horizon_states.loc['operational', 'p10']:,.0f}–"
         f"{horizon_states.loc['operational', 'p90']:,.0f}).",
)
s2.metric(
    "Expected operational GW",
    f"{sc_gw if is_scenario else op_gw_p50:,.0f} GW",
    delta=delta_gw,
    help=f"Baseline median: {op_gw_p50:,.0f} GW "
         f"(P10 {op_gw_p10:,.0f} · P90 {op_gw_p90:,.0f}).",
)
s3.metric(
    f"Projected withdrawals by {horizon_dt.year}",
    f"{sc_wd if is_scenario else expected_wd:,.0f}",
    delta=delta_wd,
    delta_color="inverse",
    help=f"Baseline mean: {expected_wd:,.0f} "
         f"(P10–P90 {horizon_states.loc['withdrawn', 'p10']:,.0f}–"
         f"{horizon_states.loc['withdrawn', 'p90']:,.0f}).",
)
s4.metric(
    "Still in queue at horizon",
    f"{sc_stuck if is_scenario else expected_stuck:,.0f}",
    delta=delta_stuck,
    delta_color="inverse",
    help="Projects that have neither reached commercial operation nor withdrawn after ten years.",
)

# Fan chart: operational GW over time (baseline always; scenario overlay if active)
quantiles = sim.operational_gw_quantiles((0.1, 0.5, 0.9))
fan_df = quantiles.reset_index().rename(columns={"month": "date"})

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=fan_df["date"], y=fan_df["p90"], line=dict(width=0),
        showlegend=False, hoverinfo="skip", name="Baseline P90",
    )
)
fig.add_trace(
    go.Scatter(
        x=fan_df["date"], y=fan_df["p10"], line=dict(width=0),
        fill="tonexty", fillcolor="rgba(150, 150, 150, 0.20)",
        name="Baseline P10–P90", hoverinfo="skip",
    )
)
baseline_color = "rgb(150, 150, 150)" if is_scenario else "rgb(99, 110, 250)"
baseline_dash = "dash" if is_scenario else "solid"
fig.add_trace(
    go.Scatter(
        x=fan_df["date"], y=fan_df["p50"],
        line=dict(color=baseline_color, width=2, dash=baseline_dash),
        name="Baseline (P50)",
        hovertemplate="Baseline · %{x|%b %Y}: %{y:,.0f} GW<extra></extra>",
    )
)

if scenario_sim is not None:
    sc_q = scenario_sim.operational_gw_quantiles((0.1, 0.5, 0.9)).reset_index().rename(
        columns={"month": "date"}
    )
    fig.add_trace(
        go.Scatter(
            x=sc_q["date"], y=sc_q["p90"], line=dict(width=0),
            showlegend=False, hoverinfo="skip", name="Scenario P90",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sc_q["date"], y=sc_q["p10"], line=dict(width=0),
            fill="tonexty", fillcolor="rgba(62, 196, 126, 0.18)",
            name="Scenario P10–P90", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sc_q["date"], y=sc_q["p50"],
            line=dict(color="rgb(62, 196, 126)", width=3),
            name="Scenario (P50)",
            hovertemplate="Scenario · %{x|%b %Y}: %{y:,.0f} GW<extra></extra>",
        )
    )

fig.update_layout(
    title=("Operational GW: scenario vs. baseline" if is_scenario
           else f"Operational GW from today's active cohort, {horizon_dt.year - 10} → {horizon_dt.year}"),
    xaxis_title="",
    yaxis_title="Operational GW (cumulative)",
    height=400,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# Stacked area: state composition over time (shows whichever view is active)
active_sim = scenario_sim if scenario_sim is not None else sim
share_df = active_sim.state_share_mean().reset_index().melt(
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
    title=("Cohort flow under your scenario" if is_scenario
           else "Where the cohort goes: average project counts by state over time"),
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
