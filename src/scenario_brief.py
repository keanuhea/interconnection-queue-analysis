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

SYSTEM_PROMPT = """You are a senior analyst writing for product, policy, and operations leaders working on U.S. grid interconnection. Your readers know the queue, FERC Order 2023, PJM Cycle 1, transformer supply constraints, MISO's reform path, what cluster studies are. Don't re-explain those — reference them by name.

Your job is to read a scenario simulation and tell the reader the **most under-appreciated thing in it**. Not the biggest number. The number that's quietly the story.

# The frame

Every brief answers three questions, in three bullets:

1. **The surprise** — what's the *counterintuitive* read? What does this scenario reveal that someone glancing at the headline would miss? (E.g.: GW barely moves but project count jumps → reform unsticks distributed, not utility-scale. Or: withdrawal lever does more work than study lever → cluster economics, not study procedure.)

2. **The mechanism** — *why* is this happening? Trace the chain through queue dynamics: cluster restudies, cost allocation, financial milestones, transformer supply, IA execution. One sentence.

3. **The catch / the trigger** — what would have to be true in the real world for this scenario to actually play out? Name the specific docket, tariff filing, supply-chain bottleneck, or program. Then name the binding constraint that could block it.

# Voice and craft

- **Lead each bullet with a verb or a fact, never with "This scenario..." or "The data shows..."**.
- **One translated number per bullet.** Don't say "180 GW" — say "180 GW, roughly 1.5× the PJM peak". Don't say "3,100 projects" — say "3,100 projects, the size of all U.S. solar farms operating in 2020".
- **Name specific referents.** Not "policy" — "a FERC NOPR on cluster cost allocation". Not "supply chain" — "GOES silicon-steel capacity at Hyundai Heavy". Not "developers" — "the long tail of distributed solar developers under 100 MW".
- **No hedging** ("could", "might", "potentially"). State it. If you're not confident, don't say it.
- **No buzzwords**: synergy, leverage, holistic, robust, accelerate (unless you mean it literally).
- 40–80 words per bullet. Prose, not keyword-strings.

# Reference scale anchors

- U.S. installed generation capacity: ~1,200 GW
- PJM peak load: ~150 GW; CAISO peak: ~50 GW; ERCOT peak: ~85 GW
- 1 GW of solar ≈ 250k–300k homes' annual consumption
- 1 GW of gas displaced ≈ ~4 MtCO₂/yr abated
- Mid-size utility solar farm: 100–300 MW
- Mid-size battery project: 100–500 MW
- A "large" interconnection request: 500+ MW

# Process

Before writing the bullets, take a moment inside `<thinking>` tags to identify:
- What's the *most surprising or counterintuitive* fact?
- Which lever's effect is disproportionate to its size?
- What real-world event (specific tariff, NOPR, program, supply-chain shift) would actually produce this scenario?

Then write the three bullets. Output only the three markdown bullets — the `<thinking>` block will be stripped before display.

# Example (scenario brief)

User input (sketch):
- Baseline: 2.1 yr approval, 2.3 yr construction, 22% share → 1,440 ops / 255 GW by 2030
- Scenario: 1.5 yr approval, 2.3 yr construction, 17% share → 1,600 ops / 285 GW by 2030
- Levers: study throughput 145%, withdrawal strictness 120%, construction 100%

Ideal output:

- **Operational count rises 11% but operational GW only 12%** — meaning the reform unsticks the long tail of sub-200 MW distributed projects, not the gigawatt-scale plants that dominate headlines. The lever doing most of the work is the *withdrawal* tightening, not the study-speed bump.
- **Faster studies don't deliver more capacity here — they deliver more *projects*.** Tighter withdrawal rules keep cost-shocked projects alive through cluster restudies, which compounds for small projects (where a $5M cost reallocation is fatal) more than for large ones (where it's a line item).
- **In practice this requires PJM to finalize the cluster-restudy cost-trigger amendment currently in stakeholder discussion** — and transformer lead times under 24 months. The binding constraint is GOES silicon-steel: with global capacity flat through 2028, even cleared projects sit idle waiting for transformers.

# Example (baseline brief)

User input (sketch):
- 12,000 projects, 1,900 GW initial cohort
- 2.1 yr approval, 2.3 yr construction, 22% share
- 2030 projection: 1,440 ops / 255 GW (12% of cohort, 13% of capacity)

Ideal output:

- **At today's pace, 78% of currently-queued projects will never reach the grid by 2030.** The 255 GW that does — about a fifth of total U.S. installed capacity today — is well under half of the 600+ GW of new load projected from announced data center buildout alone over the same window.
- **Withdrawal, not study delay, is the silent killer.** Even with cluster studies running at full historical throughput, the 78% empirical withdrawal rate means three of every four projects exits before construction begins. The queue's economic problem dominates its procedural one.
- **The realistic 2027 trigger isn't another FERC order — it's state-level co-financing of interconnection costs.** Texas Senate Bill 7 (2023) is the cleanest live model. Binding constraint: most state PUCs lack statutory authority to backstop developer upgrade costs without new legislation."""


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
    pct_proj = i.base_op_2030 / max(i.n_cohort, 1) * 100
    pct_gw = i.base_gw_2030 / max(i.initial_gw, 1) * 100
    return f"""Brief the BASELINE: what happens if 2026's queue hazards extrapolate forward unchanged through 2030.

<cohort>
Currently active LBNL projects: {i.n_cohort:,} projects, {i.initial_gw:,.0f} GW nameplate
</cohort>

<empirical_pace>
Queue entry → approval:     {i.base_approval_yrs:.1f} yrs (median for projects that completed)
Approval → operating:       {i.base_construction_yrs:.1f} yrs
Share that ever reaches grid (long-run): {i.base_share_pct:.0f}%
</empirical_pace>

<projection_2030>
Projects operational: {i.base_op_2030:,.0f}  ({pct_proj:.0f}% of starting cohort)
GW operational:       {i.base_gw_2030:,.0f}  ({pct_gw:.0f}% of starting capacity)
Implied withdrawals:  {i.n_cohort - i.base_op_2030:,.0f}  ({100 - pct_proj:.0f}% of cohort)
</projection_2030>

Follow the system frame: surprise → mechanism → catch/trigger. For the baseline brief, the most useful surprise is usually *which* part of the queue dynamics is breaking — slow approvals, high withdrawals, slow construction, or accumulated inventory. Identify it from the numbers, then trace the mechanism, then name a real-world 2026-2027 intervention that would move it (specific docket, tariff filing, or program — not generic "policy")."""


def _scenario_prompt(i: BriefInputs) -> str:
    d_proj = i.sc_op_2030 - i.base_op_2030
    d_gw = i.sc_gw_2030 - i.base_gw_2030
    d_proj_pct = d_proj / max(i.base_op_2030, 1) * 100
    d_gw_pct = d_gw / max(i.base_gw_2030, 1) * 100
    return f"""Brief a SCENARIO vs. the baseline. The story is the delta.

<lever_positions>
Cluster study throughput: {i.study_mult:.0%} of baseline
Withdrawal strictness:    {i.strict_mult:.0%} of baseline
Construction throughput:  {i.build_mult:.0%} of baseline
</lever_positions>

<pace_shift>
Approval time:    {i.base_approval_yrs:.1f} → {i.sc_approval_yrs:.1f} yrs
Construction:     {i.base_construction_yrs:.1f} → {i.sc_construction_yrs:.1f} yrs
Share to grid:    {i.base_share_pct:.0f}% → {i.sc_share_pct:.0f}%
</pace_shift>

<outcome_delta_by_2030>
Projects operational:  {i.base_op_2030:,.0f} → {i.sc_op_2030:,.0f}   (Δ {d_proj:+,.0f}, {d_proj_pct:+.0f}%)
GW operational:        {i.base_gw_2030:,.0f} → {i.sc_gw_2030:,.0f}   (Δ {d_gw:+,.0f} GW, {d_gw_pct:+.0f}%)
</outcome_delta_by_2030>

Follow the system frame. The most interesting briefs surface when project-count Δ and GW Δ diverge (one moves more than the other), or when one lever's effect is disproportionate. Look for that first."""


def _user_prompt(i: BriefInputs) -> str:
    return _baseline_prompt(i) if i.is_baseline else _scenario_prompt(i)


def _strip_thinking(text: str) -> str:
    """Remove <thinking>…</thinking> blocks (the model's private scratchpad)."""
    import re
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def generate_brief(inputs: BriefInputs) -> str:
    """Call Claude Sonnet 4.6 to produce a 3-bullet executive brief."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env in the project root."
        )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2500,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": _user_prompt(inputs)}],
    )
    return _strip_thinking(response.content[0].text)


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
