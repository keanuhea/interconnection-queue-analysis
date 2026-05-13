"""Generate executive briefs of scenario outcomes via the Anthropic API.

Wired up to the operator-side lever panel: the user pulls sliders, the
simulator runs, and Claude writes a 3-bullet exec read of what just happened.
The brief is grounded in concrete numbers from the simulation (deltas in
projects + GW operational, the lever positions, real-world policy context).

The system prompt is cached so repeated calls amortize. Each generation costs
a few cents at Sonnet 4.6 pricing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an executive-briefing analyst for U.S. grid infrastructure policy. You write tight, concrete, numerate briefs for senior product and policy leaders.

Your readers are operators, policy leads, and product executives who already understand the interconnection queue: they know FERC Order 2023, PJM's Cycle 1, the transformer supply chain story, what cluster studies are. Don't explain those — reference them.

Style:
- Don't hedge. Don't repeat input numbers back at readers — interpret them.
- Be concrete with numbers in your conclusions.
- No preamble, no closing summary. Just three bullets.

Format: exactly three markdown bullets. Each ~30 words. The bullets should land:
1. The headline impact in plain language. Lead with the biggest number.
2. Which operator-side lever is doing most of the work, and why.
3. The real-world policy or market move that would actually produce this scenario, plus the binding constraint that could block it."""


@dataclass(frozen=True)
class BriefInputs:
    base_approval_yrs: float
    base_construction_yrs: float
    base_share_pct: float
    sc_approval_yrs: float
    sc_construction_yrs: float
    sc_share_pct: float
    study_mult: float
    strict_mult: float
    build_mult: float
    base_op_2030: float
    base_gw_2030: float
    sc_op_2030: float
    sc_gw_2030: float
    n_cohort: int = 0
    initial_gw: float = 0.0
    is_baseline: bool = False


def _baseline_prompt(i: BriefInputs) -> str:
    return f"""Generate a 3-bullet executive brief about the BASELINE scenario — what happens if today's hazards extrapolate forward unchanged.

Starting cohort (currently active LBNL projects):
- {i.n_cohort:,} projects, {i.initial_gw:,.0f} GW of nameplate capacity

Today's operating pace (empirical medians for projects that completed each transition):
- Queue entry → approval: {i.base_approval_yrs:.1f} yrs
- Approval → operating:   {i.base_construction_yrs:.1f} yrs
- Long-run share of new entries reaching the grid: {i.base_share_pct:.0f}%

Projected by 2030 if today's pace continues (500-replicate Monte Carlo):
- Projects operational: {i.base_op_2030:,.0f}  ({i.base_op_2030 / max(i.n_cohort, 1) * 100:.0f}% of starting cohort)
- GW operational:       {i.base_gw_2030:,.0f}  ({i.base_gw_2030 / max(i.initial_gw, 1) * 100:.0f}% of starting capacity)

Write three bullets per the system instructions. For the baseline brief:
1. State plainly what the baseline projection means — the headline impact of doing nothing. Lead with the biggest number.
2. Identify which part of the queue dynamics is most responsible for the slow clearance (slow approvals, high withdrawal rate, slow construction, or accumulated stuck inventory).
3. Name the realistic intervention that would move this — what specific policy or market change would meaningfully change the projection, and what's the binding constraint."""


def _scenario_prompt(i: BriefInputs) -> str:
    return f"""Generate a 3-bullet executive brief for this interconnection-queue scenario vs. baseline.

Baseline (today's pace):
- Avg queue entry → approval: {i.base_approval_yrs:.1f} yrs
- Avg approval → operating:   {i.base_construction_yrs:.1f} yrs
- Long-run share reaching the grid: {i.base_share_pct:.0f}%

Scenario:
- Avg queue entry → approval: {i.sc_approval_yrs:.1f} yrs
- Avg approval → operating:   {i.sc_construction_yrs:.1f} yrs
- Long-run share reaching the grid: {i.sc_share_pct:.0f}%

Underlying operator-side lever positions:
- Cluster study throughput: {i.study_mult:.0%} of baseline
- Withdrawal strictness:    {i.strict_mult:.0%} of baseline
- Construction throughput:  {i.build_mult:.0%} of baseline

Cohort outcomes by 2030 (500-replicate Monte Carlo, baseline vs scenario):
- Projects operational: {i.base_op_2030:,.0f} → {i.sc_op_2030:,.0f}  (Δ {i.sc_op_2030 - i.base_op_2030:+,.0f})
- GW operational:       {i.base_gw_2030:,.0f} → {i.sc_gw_2030:,.0f}  (Δ {i.sc_gw_2030 - i.base_gw_2030:+,.0f})

Write three bullets per the system instructions."""


def _user_prompt(i: BriefInputs) -> str:
    return _baseline_prompt(i) if i.is_baseline else _scenario_prompt(i)


def generate_brief(inputs: BriefInputs) -> str:
    """Call Claude Sonnet 4.6 to produce a 3-bullet executive brief."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env in the project root."
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": _user_prompt(inputs)}],
    )
    return response.content[0].text


if __name__ == "__main__":
    test = BriefInputs(
        base_approval_yrs=2.2,
        base_construction_yrs=2.0,
        base_share_pct=21.0,
        sc_approval_yrs=1.5,
        sc_construction_yrs=2.0,
        sc_share_pct=17.0,
        study_mult=1.45,
        strict_mult=1.20,
        build_mult=1.00,
        base_op_2030=2400,
        base_gw_2030=180,
        sc_op_2030=3100,
        sc_gw_2030=240,
    )
    print(generate_brief(test))
