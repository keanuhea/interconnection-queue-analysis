"""Monte Carlo forward simulation of queue progression.

Given a current cohort (one row per active project, with state + MW) and a
fitted HazardTable, simulate K trajectories N years forward at monthly
resolution. At each step every project either stays put or transitions to the
next state, sampled from the hazard probabilities.

Returns a `SimResult` with two views the dashboard cares about:
- `state_counts[k, t, state]` — projects in each state at month t, replicate k
- `gw_operational[k, t]` — total operational GW at month t, replicate k
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.state_machine import (
    ACTIVE_STATES,
    HazardTable,
    State,
    cohort_from_lbnl,
    fit_hazards,
)
from src.load_data import load_queued_up

ALL_STATES = (State.SUBMITTED, State.IA_SIGNED, State.OPERATIONAL, State.WITHDRAWN)
STATE_INDEX = {s: i for i, s in enumerate(ALL_STATES)}


@dataclass
class SimResult:
    """Output of `simulate()`. All arrays indexed [replicate, month_offset, ...]."""

    months: pd.DatetimeIndex
    states: tuple[State, ...]
    state_counts: np.ndarray  # shape (K, T, |states|)
    gw_operational: np.ndarray  # shape (K, T)
    initial_mw: np.ndarray  # shape (n_projects,)
    initial_states: list[State]
    asof: pd.Timestamp

    def operational_gw_quantiles(self, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        out = {}
        for q in qs:
            out[f"p{int(q * 100)}"] = np.quantile(self.gw_operational, q, axis=0)
        df = pd.DataFrame(out, index=self.months)
        df.index.name = "month"
        return df

    def state_share_mean(self) -> pd.DataFrame:
        mean = self.state_counts.mean(axis=0)
        df = pd.DataFrame(mean, index=self.months, columns=[s.value for s in self.states])
        df.index.name = "month"
        return df

    def state_at_horizon(self, horizon_months: int) -> pd.DataFrame:
        """Per-state count distribution at a specific month offset."""
        idx = min(horizon_months, len(self.months) - 1)
        slice_ = self.state_counts[:, idx, :]
        means = slice_.mean(axis=0)
        p10 = np.quantile(slice_, 0.1, axis=0)
        p90 = np.quantile(slice_, 0.9, axis=0)
        return pd.DataFrame(
            {"mean": means, "p10": p10, "p90": p90},
            index=[s.value for s in self.states],
        )


def _build_transition_matrix(
    table: HazardTable, scenario_multipliers: dict | None = None
) -> np.ndarray:
    """Return a |states| x |states| row-stochastic monthly transition matrix.

    `scenario_multipliers[(from_state, to_state)]` scales that transition's
    monthly probability. Used for what-if scenarios; pass None for baseline.
    Stay-probability is recomputed after multipliers so rows still sum to 1.
    """
    multipliers = scenario_multipliers or {}
    n = len(ALL_STATES)
    m = np.zeros((n, n))

    for s in ALL_STATES:
        i = STATE_INDEX[s]
        if s not in ACTIVE_STATES:
            m[i, i] = 1.0
            continue
        out_probs = {}
        for to_state, p in table.monthly_p.get(s, {}).items():
            scaled = p * float(multipliers.get((s, to_state), 1.0))
            out_probs[to_state] = max(0.0, min(1.0, scaled))
        total_out = sum(out_probs.values())
        if total_out > 1.0:
            scale = 1.0 / total_out
            out_probs = {k: v * scale for k, v in out_probs.items()}
            total_out = 1.0
        m[i, i] = 1.0 - total_out
        for to_state, p in out_probs.items():
            m[i, STATE_INDEX[to_state]] = p

    return m


def simulate(
    cohort: pd.DataFrame,
    table: HazardTable,
    horizon_years: float = 10.0,
    n_replicates: int = 500,
    scenario_multipliers: dict | None = None,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Run forward Monte Carlo from `cohort`'s current states.

    Vectorized: at each month, every project samples its next state from a
    categorical distribution row-indexed by current state. The transition
    matrix is constant across time and projects (the modeling caveat is that
    we ignore feature-conditional hazards for now — uniformity per state).
    """
    rng = rng or np.random.default_rng(42)
    n = len(cohort)
    if n == 0:
        raise ValueError("cohort is empty — nothing to simulate.")

    states = ALL_STATES
    n_states = len(states)
    months_total = int(round(horizon_years * 12)) + 1

    transition_matrix = _build_transition_matrix(table, scenario_multipliers)
    cumulative = transition_matrix.cumsum(axis=1)

    initial_idx = np.array([STATE_INDEX[s] for s in cohort["state"]], dtype=np.int8)
    mw = cohort["mw"].fillna(0).to_numpy(dtype=np.float64)

    K = n_replicates
    current = np.tile(initial_idx, (K, 1))  # shape (K, n)

    state_counts = np.zeros((K, months_total, n_states), dtype=np.int32)
    gw_operational = np.zeros((K, months_total), dtype=np.float64)
    op_idx = STATE_INDEX[State.OPERATIONAL]

    for t in range(months_total):
        for s_idx in range(n_states):
            mask = current == s_idx
            state_counts[:, t, s_idx] = mask.sum(axis=1)
        operational_mask = (current == op_idx)
        gw_operational[:, t] = (operational_mask * mw).sum(axis=1) / 1000.0

        if t == months_total - 1:
            break
        u = rng.random((K, n))
        cum = cumulative[current]  # shape (K, n, n_states)
        next_idx = (u[..., None] >= cum).sum(axis=-1).astype(np.int8)
        next_idx = np.clip(next_idx, 0, n_states - 1)
        current = next_idx

    months = pd.date_range(table.asof, periods=months_total, freq="MS")

    return SimResult(
        months=months,
        states=states,
        state_counts=state_counts,
        gw_operational=gw_operational,
        initial_mw=mw,
        initial_states=list(cohort["state"]),
        asof=table.asof,
    )


def summarize(result: SimResult) -> None:
    print(f"Replicates: {result.gw_operational.shape[0]:,}")
    print(f"Horizon:    {len(result.months) - 1} months "
          f"({result.months[0].date()} → {result.months[-1].date()})")
    print(f"Initial cohort: {len(result.initial_mw):,} projects, "
          f"{result.initial_mw.sum() / 1000:,.0f} GW")
    print()
    print("Operational GW over time (P10 / P50 / P90):")
    q = result.operational_gw_quantiles()
    print(q.iloc[::12].round(1).to_string())
    print()

    horizon_months = len(result.months) - 1
    horizon = result.state_at_horizon(horizon_months)
    print(f"\nState counts at horizon ({result.months[-1].date()}):")
    print(horizon.round(1).to_string())


if __name__ == "__main__":
    df = load_queued_up()
    table = fit_hazards(df)
    cohort = cohort_from_lbnl(df, table.asof)
    print(f"Active cohort to simulate: {len(cohort):,} projects\n")

    result = simulate(cohort, table, horizon_years=10, n_replicates=500)
    summarize(result)
