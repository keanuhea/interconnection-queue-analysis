"""Multi-state model of interconnection queue progression.

Projects move through a chain of milestones: queue entry → IA signed → operational.
At any point, a project can be withdrawn (absorbing). We treat this as a
continuous-time Markov chain with piecewise-constant hazards estimated empirically
from the LBNL resolved cohort.

States (coarse, supported directly by LBNL date columns):

    SUBMITTED ─────► IA_SIGNED ─────► OPERATIONAL
        │               │
        ▼               ▼
    WITHDRAWN       WITHDRAWN

The finer PJM milestone chain (Feasibility → SIS → Facilities → IA) requires
multiple snapshots to observe transitions and is left as a follow-up; the same
hazard-estimation machinery extends to it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from src.load_data import load_queued_up


class State(str, Enum):
    SUBMITTED = "submitted"
    IA_SIGNED = "ia_signed"
    OPERATIONAL = "operational"
    WITHDRAWN = "withdrawn"


ACTIVE_STATES = (State.SUBMITTED, State.IA_SIGNED)
ABSORBING_STATES = (State.OPERATIONAL, State.WITHDRAWN)


@dataclass(frozen=True)
class Transition:
    """A directed move between two states. `from_state -> to_state`."""

    from_state: State
    to_state: State

    def __str__(self) -> str:
        return f"{self.from_state.value} → {self.to_state.value}"


CANONICAL_TRANSITIONS = (
    Transition(State.SUBMITTED, State.IA_SIGNED),
    Transition(State.SUBMITTED, State.WITHDRAWN),
    Transition(State.IA_SIGNED, State.OPERATIONAL),
    Transition(State.IA_SIGNED, State.WITHDRAWN),
)


def derive_current_state(row: pd.Series, asof: pd.Timestamp) -> State:
    """Classify a project's state as of a reference date.

    Order of precedence: withdrawn > operational > IA signed > submitted.
    Withdrawal and operational are absorbing — once entered, we don't backtrack.
    """
    wd = row.get("withdrawn_date")
    if pd.notna(wd) and wd <= asof:
        return State.WITHDRAWN
    op = row.get("operational_date")
    if pd.notna(op) and op <= asof:
        return State.OPERATIONAL
    ia = row.get("ia_signed")
    if pd.notna(ia) and ia <= asof:
        return State.IA_SIGNED
    return State.SUBMITTED


def _exit_observation(row: pd.Series, from_state: State) -> tuple[pd.Timestamp | None, State | None]:
    """Return (exit_time, exit_state) for a project starting in `from_state`.

    Returns (None, None) if the project never left `from_state` in the dataset
    (right-censored). The exit_state is the state the project actually moved
    *into* — used to attribute the transition to the correct hazard.
    """
    if from_state == State.SUBMITTED:
        ia = row.get("ia_signed")
        wd = row.get("withdrawn_date")
        op = row.get("operational_date")

        events = []
        if pd.notna(ia):
            events.append((ia, State.IA_SIGNED))
        if pd.notna(wd):
            events.append((wd, State.WITHDRAWN))
        if pd.notna(op) and pd.isna(ia):
            events.append((op, State.IA_SIGNED))

        if not events:
            return None, None
        events.sort()
        return events[0]

    if from_state == State.IA_SIGNED:
        ia = row.get("ia_signed")
        if pd.isna(ia):
            return None, None
        op = row.get("operational_date")
        wd = row.get("withdrawn_date")

        events = []
        if pd.notna(op) and op > ia:
            events.append((op, State.OPERATIONAL))
        if pd.notna(wd) and wd > ia:
            events.append((wd, State.WITHDRAWN))

        if not events:
            return None, None
        events.sort()
        return events[0]

    return None, None


def _entry_time(row: pd.Series, state: State) -> pd.Timestamp | None:
    if state == State.SUBMITTED:
        return row.get("queue_date")
    if state == State.IA_SIGNED:
        return row.get("ia_signed")
    return None


@dataclass
class HazardTable:
    """Monthly transition probabilities, indexed by (from_state, to_state).

    `monthly_p[from][to]` is the probability that a project in `from` moves to
    `to` within a one-month window. We assume piecewise-constant hazards over
    the calibration window — coarse, but defensible given LBNL coverage.

    `n_observed[from][to]` records the number of empirical transitions used to
    fit the rate (so the dashboard can flag thinly-fit cells).
    """

    monthly_p: dict[State, dict[State, float]]
    n_observed: dict[State, dict[State, int]]
    asof: pd.Timestamp

    def stay_prob(self, state: State) -> float:
        if state in ABSORBING_STATES:
            return 1.0
        return max(0.0, 1.0 - sum(self.monthly_p.get(state, {}).values()))


def fit_hazards(
    df: pd.DataFrame,
    asof: pd.Timestamp | None = None,
    min_year: int = 2010,
) -> HazardTable:
    """Estimate per-month transition probabilities from the LBNL cohort.

    For each canonical transition, we sum (a) the number of months projects
    spent at risk in the from-state and (b) the count of transitions actually
    observed into the to-state. The monthly hazard is count / exposure.

    `min_year` filters out very old queue entries that distort the rate (LBNL
    has some pre-2003 sentinel entries).
    """
    asof = asof or pd.Timestamp("2024-12-31")
    df = df[df["queue_date"].dt.year >= min_year].copy()

    counts: dict[State, dict[State, int]] = {s: {} for s in ACTIVE_STATES}
    exposures: dict[State, float] = {s: 0.0 for s in ACTIVE_STATES}

    for _, row in df.iterrows():
        for from_state in ACTIVE_STATES:
            entry = _entry_time(row, from_state)
            if pd.isna(entry) or entry > asof:
                continue
            exit_time, exit_state = _exit_observation(row, from_state)
            end = min(exit_time, asof) if exit_time is not None else asof
            months = max(0.0, (end - entry).days / 30.4375)
            exposures[from_state] += months
            if exit_state is not None and exit_time is not None and exit_time <= asof:
                counts[from_state][exit_state] = counts[from_state].get(exit_state, 0) + 1

    monthly_p: dict[State, dict[State, float]] = {s: {} for s in ACTIVE_STATES}
    n_observed: dict[State, dict[State, int]] = {s: {} for s in ACTIVE_STATES}
    for tr in CANONICAL_TRANSITIONS:
        n = counts[tr.from_state].get(tr.to_state, 0)
        exp_months = exposures[tr.from_state]
        rate = (n / exp_months) if exp_months > 0 else 0.0
        monthly_p[tr.from_state][tr.to_state] = float(rate)
        n_observed[tr.from_state][tr.to_state] = int(n)

    return HazardTable(monthly_p=monthly_p, n_observed=n_observed, asof=asof)


def cohort_from_lbnl(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Snapshot: every still-active LBNL project with its current state.

    The `state` column stores the string `.value` (not the State enum). This
    avoids dtype-downcast surprises when the DataFrame is round-tripped through
    streamlit's Arrow-backed cache, which strips Enum identity from `str+Enum`
    subclasses. Use State(row["state"]) to recover the enum when needed.
    """
    snap = df.copy()
    snap["state"] = snap.apply(lambda r: derive_current_state(r, asof).value, axis=1)
    snap["state_value"] = snap["state"]
    snap["months_in_state"] = snap.apply(
        lambda r: _months_in_state(r, State(r["state"]), asof), axis=1
    )
    active_values = [s.value for s in ACTIVE_STATES]
    active = snap[snap["state"].isin(active_values)].copy()
    return active


def _months_in_state(row: pd.Series, state: State, asof: pd.Timestamp) -> float:
    entry = _entry_time(row, state)
    if pd.isna(entry):
        return 0.0
    return max(0.0, (asof - entry).days / 30.4375)


def summarize(table: HazardTable) -> None:
    print(f"Fit as of: {table.asof.date()}\n")
    print(f"{'transition':<35} {'monthly P':>10}  {'~yearly P':>10}  {'n':>6}")
    print("-" * 70)
    for tr in CANONICAL_TRANSITIONS:
        p = table.monthly_p[tr.from_state][tr.to_state]
        py = 1 - (1 - p) ** 12
        n = table.n_observed[tr.from_state][tr.to_state]
        print(f"{str(tr):<35} {p:>10.4f}  {py:>10.4f}  {n:>6}")
    for s in ACTIVE_STATES:
        print(f"{('stay in ' + s.value):<35} {table.stay_prob(s):>10.4f}")


if __name__ == "__main__":
    df = load_queued_up()
    table = fit_hazards(df)
    summarize(table)

    print("\nCurrent active cohort (as of 2024-12-31):")
    cohort = cohort_from_lbnl(df, table.asof)
    print(cohort["state_value"].value_counts().to_string())
