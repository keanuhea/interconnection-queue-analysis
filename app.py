"""Streamlit dashboard for the Queued Up interconnection analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.concentration_analysis import top_concentration
from src.load_data import load_queued_up
from src.withdrawal_model import score_open_queue, train

st.set_page_config(page_title="Interconnection Queue Analysis", layout="wide")


@st.cache_data(show_spinner="Loading Queued Up dataset...")
def _load() -> pd.DataFrame:
    return load_queued_up()


@st.cache_resource(show_spinner="Training withdrawal model...")
def _train(df: pd.DataFrame):
    clf, encoder, importances = train(df)
    return clf, encoder, importances


st.title("U.S. Interconnection Queue Analysis")
st.caption(
    "Berkeley Lab Queued Up 2025 dataset · queue concentration, withdrawal "
    "probability, and corridor risk."
)

try:
    df = _load()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

total_mw = df["mw"].sum() if "mw" in df.columns else None
col1, col2, col3, col4 = st.columns(4)
col1.metric("Projects in dataset", f"{len(df):,}")
col2.metric("Withdrawn (historic)", f"{df['withdrawn'].mean():.0%}")
col3.metric("Operational", f"{df['operational'].mean():.0%}")
if total_mw is not None:
    col4.metric("Total queued MW", f"{total_mw/1000:,.0f} GW")

st.divider()

tab_overview, tab_concentration, tab_model = st.tabs(
    ["Queue overview", "POI concentration", "Withdrawal model"]
)

with tab_overview:
    st.subheader("Queue size by RTO over time")
    if "queue_date" in df.columns and "rto" in df.columns:
        plot_df = df.dropna(subset=["queue_date", "rto"]).copy()
        plot_df["year"] = plot_df["queue_date"].dt.year
        agg = (
            plot_df.groupby(["year", "rto"])
            .size()
            .reset_index(name="projects")
        )
        fig = px.area(agg, x="year", y="projects", color="rto",
                      title="Projects entering queue per year, by RTO")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("queue_date or rto columns not detected.")

    if "resource_type" in df.columns and "mw" in df.columns:
        st.subheader("Queued MW by resource type")
        rt = (
            df.groupby("resource_type")["mw"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(rt, x="resource_type", y="mw",
                     title="Total MW in queue by resource type")
        st.plotly_chart(fig, use_container_width=True)

with tab_concentration:
    st.subheader("Top 20 highest-concentration POIs")
    n = st.slider("How many POIs to show", 5, 50, 20)
    top = top_concentration(df, n)
    st.dataframe(top, use_container_width=True)

    if not top.empty:
        fig = px.bar(
            top.iloc[::-1],
            x="total_mw",
            y="poi",
            color="project_count",
            orientation="h",
            title=f"Top {n} POIs by total queued MW",
        )
        fig.update_layout(height=max(400, 22 * n))
        st.plotly_chart(fig, use_container_width=True)

with tab_model:
    st.subheader("Withdrawal probability model")
    st.caption(
        "Gradient boosting classifier trained on resolved projects (withdrawn "
        "vs. operational). Predictions below are for currently in-progress projects."
    )
    try:
        clf, encoder, importances = _train(df)
    except ValueError as e:
        st.error(str(e))
    else:
        st.write("**Top feature importances**")
        st.bar_chart(importances)

        scored = score_open_queue(df, clf, encoder)
        if not scored.empty and "resource_type" in scored.columns:
            fig = px.box(
                scored,
                x="resource_type",
                y="p_withdraw",
                title="Predicted P(withdraw) by resource type, in-progress queue",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                scored[["project_name", "rto", "resource_type", "mw",
                        "queue_age_years", "p_withdraw"]]
                .sort_values("p_withdraw", ascending=False)
                .head(50),
                use_container_width=True,
            )
