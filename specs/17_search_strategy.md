# Spec 17: Experiment Search Strategy

## Problem

The naive approach is grid search over all dimensions. That's 5 sizes x
5 mixtures x 5 recipes x 4 curricula = 500 runs. Way too expensive.

We need a smarter search that finds the best configurations in fewer runs,
while still producing valid scaling curves.

---

## Search Principles

1. **One variable at a time.** Hold everything else fixed, sweep one dimension.
   This gives clean comparisons and avoids confounding.

2. **Start in the middle, then branch.** Run the M (130M) model first.
   It's big enough to show effects but small enough to iterate quickly.

3. **Eliminate early.** If a mixture/recipe is clearly worse at M size,
   don't run it at other sizes. Save compute for winners.

4. **Binary search for thresholds.** When we're looking for "at what size
   does capability X emerge?", don't sweep all sizes — binary search.

5. **Exploit structure.** Scaling laws are smooth. If performance at 50M and
   130M is known, we can predict 85M. Only verify surprising predictions.

---

## Phase 1: Pretrain Search

### Strategy: Pilot → Eliminate → Sweep

```
Step 1: Run 3 mixtures at M size (3 runs)
        → Result: ranked list of mixtures

Step 2: Is #1 clearly better than #2? (CIs don't overlap)
        YES → Drop #3 and below. Run #1 at XS and XL (2 runs)
              → Now have 3 points on the scaling curve
              → Interpolate S and L. Do they match prediction?
              → If yes: predicted curve is good, fill in S and L (2 runs) for completeness
              → If no: something interesting at that scale. Run S and L (2 runs)
        NO  → Mixtures might interact with scale. Run top 2 at XS and XL (4 runs)
              → Check if ranking flips at different scales

Step 3: Token budget search at M size (3 runs: 20x, 50x, 100x)
        → Is 50x the sweet spot? Or does 100x still help?
        → Binary search: if 100x >> 50x, try 75x. If 50x ≈ 100x, try 30x.
```

Total: 8-12 runs (vs 15+ for full grid).

### Decision Tree

```
                    Run 3 mixtures at M
                    ┌────────┼────────┐
                    │        │        │
              mix-general  mix-broad  mix-heavy
                    │        │        │
                    └────────┼────────┘
                         Rank them
                    ┌────────┴────────┐
               Clear winner?     Too close to call?
                    │                 │
            Run winner at          Run top 2 at
            XS, XL (2 runs)       XS, XL (4 runs)
                    │                 │
            Scaling curve          Scale-dependent?
            smooth?                     │
             │    │              ┌──────┴──────┐
           Yes    No          Yes: both       No: pick
             │    │           matter          winner
           Fill  Investigate        │              │
           S, L  S, L         Run both         Run winner
           (2)   (2)          all sizes        all sizes
```

---

## Phase 2: SFT Search

### Strategy: Tournament → Best of Breed

```
Step 1: Run all 5 SFT recipes at M size with best pretrain (5 runs)
        → Rank recipes by GSM8K pass@1

Step 2: Statistical significance test between top 2
        → If significant: winner goes forward. Run at all sizes (4 runs)
        → If not significant: run both at S and L (4 runs), see if ranking holds

Step 3: LR sensitivity check (only if results look suboptimal)
        → Take winning recipe at M, try 4 LRs (4 runs)
        → Binary search: if 2e-5 is best, try 1.5e-5 and 3e-5

Step 4: Interaction check (optional, 3 runs)
        → Run winning recipe on different pretrain bases for M
        → Does pretrain mixture affect SFT outcome?
```

Total: 9-16 runs (vs 25+ for full grid).

### Recipe Elimination Logic

```python
def should_eliminate_recipe(results_at_m: dict) -> list[str]:
    """Decide which recipes to drop after Phase 2a."""
    ranked = sorted(results_at_m.items(), key=lambda x: x[1]["gsm8k_pass1"], reverse=True)

    keep = [ranked[0][0]]  # always keep the best

    for i in range(1, len(ranked)):
        recipe = ranked[i][0]
        score = ranked[i][1]["gsm8k_pass1"]
        best_score = ranked[0][1]["gsm8k_pass1"]
        ci_width = ranked[i][1]["gsm8k_ci95_width"]

        # Keep if within 1 CI of the best
        if best_score - score < ci_width:
            keep.append(recipe)
        # Keep if it's interesting for other reasons (e.g., much cheaper)
        elif ranked[i][1]["cost_usd"] < ranked[0][1]["cost_usd"] * 0.5:
            keep.append(recipe)  # cost-efficient alternative

    eliminated = [r[0] for r in ranked if r[0] not in keep]
    return keep, eliminated
```

---

## Phase 3: RL Search

### Strategy: Feasibility → Curriculum Comparison

```
Step 1: Sanity check — does RL help at ALL for M size? (1 run)
        → Run easy→hard curriculum on best SFT-M checkpoint
        → If no improvement after 500 steps: investigate before continuing
        → If improvement: proceed

Step 2: Curriculum comparison at M size (3 runs for remaining curricula)
        → Compare: easy→hard, hard-only, interleaved, reverse
        → Pick winner

Step 3: Run winning curriculum at all sizes (4 runs)
        → Fill out the full pipeline scaling curve

Step 4: Hyperparameter sensitivity (optional, only if RL gains are small)
        → Binary search on KL penalty: {0.01, 0.05, 0.1}
        → If 0.05 is best, try 0.03 and 0.07
```

Total: 8-12 runs.

---

## Binary Search for Capability Thresholds

### Use Case: "At what size does GSM8K > 20% emerge?"

If XS=5% and M=34%, the threshold is between XS and M.

```python
def binary_search_threshold(
    metric: str,
    threshold: float,
    known_points: dict[int, float],  # depth → score
    available_depths: list[int],
) -> int:
    """Find the smallest depth that achieves the threshold."""

    # Sort by depth
    sorted_depths = sorted(known_points.keys())

    # Find bracket
    below = max([d for d in sorted_depths if known_points[d] < threshold], default=None)
    above = min([d for d in sorted_depths if known_points[d] >= threshold], default=None)

    if below is None or above is None:
        return None  # threshold is outside our range

    # Binary search between below and above
    candidates = [d for d in available_depths if below < d < above]
    if not candidates:
        return above  # no intermediate sizes to test

    mid = candidates[len(candidates) // 2]
    return mid  # "Run this depth next to narrow the bracket"
```

### Use Case: "What's the minimum data to achieve X?"

Binary search on token multiplier:
```
20x → GSM8K 0.10
50x → GSM8K 0.15
100x → GSM8K 0.16

→ Most gain between 20x and 50x. Try 35x.
35x → GSM8K 0.14

→ Diminishing returns after ~40x. The sweet spot is 40-50x.
```

---

## Agent Search Protocol

The agent follows this protocol to decide what to run next:

```python
def propose_next_experiments(state: ExperimentState) -> list[Experiment]:
    """Agent uses this to decide what to run next."""

    proposals = []

    if state.current_phase == "pretrain":
        if not state.has_pilot_results():
            # Step 1: pilot
            proposals = generate_pilot_runs(state)
        elif not state.has_mixture_ranking():
            # Step 2: analyze pilot, decide on elimination
            ranking = rank_mixtures(state.pilot_results)
            keep, drop = should_eliminate(ranking)
            proposals = generate_scaling_runs(keep, state)
        elif not state.has_token_budget_data():
            # Step 3: token budget sweep
            proposals = generate_token_sweep(state)
        else:
            # Done with pretrain search
            proposals = []

    elif state.current_phase == "sft":
        if not state.has_recipe_comparison():
            proposals = generate_recipe_comparison(state)
        elif not state.has_sft_scaling_curve():
            winner = pick_winning_recipe(state)
            proposals = generate_sft_scaling(winner, state)
        else:
            proposals = []

    elif state.current_phase == "grpo":
        if not state.has_rl_feasibility():
            proposals = [generate_rl_feasibility_run(state)]
        elif not state.has_curriculum_comparison():
            proposals = generate_curriculum_comparison(state)
        elif not state.has_rl_scaling_curve():
            winner = pick_winning_curriculum(state)
            proposals = generate_rl_scaling(winner, state)
        else:
            proposals = []

    # Annotate proposals with cost estimates and rationale
    for p in proposals:
        p.estimated_cost = estimate_cost(p)
        p.rationale = explain_why(p, state)

    return proposals
```

### Agent Proposal Format

```markdown
## Proposed Experiments (Wave 4)

Based on Phase 2a results (recipe comparison at M):
- concise-cot: GSM8K 34% (±3%)
- quality: GSM8K 32% (±3%)
- kitchen-sink: GSM8K 31% (±3%)
- distill-r1: GSM8K 28% (±3%)
- progressive: GSM8K 27% (±4%)

**Analysis:** concise-cot and quality are within each other's CIs.
distill-r1 and progressive are clearly worse — eliminating them.

**Proposal:** Run concise-cot and quality on S (85M) and L (200M)
to check if ranking holds across scale.

| Run | Depth | Recipe | Est. Cost | Est. Time |
|-----|-------|--------|-----------|-----------|
| sft-s-concise | 12 | concise-cot | ~$5 | ~1h |
| sft-s-quality | 12 | quality | ~$5 | ~1h |
| sft-l-concise | 20 | concise-cot | ~$7 | ~2h |
| sft-l-quality | 20 | quality | ~$7 | ~2h |

**Total: ~$24, ~2h (parallel)**

**Decision needed:** Approve these 4 runs? Or would you prefer
to just run concise-cot everywhere and skip the comparison?
```

---

## Stopping Rules

### When to Stop Searching

A dimension is "resolved" when:
1. Winner is statistically significant (CIs don't overlap), OR
2. Top N configs are within noise — pick any (declare tie, save compute), OR
3. Budget for this phase is exhausted

### When to Add Runs

Add runs when:
1. Results are surprising (contradicts hypothesis → need more data)
2. Two configs are very close (need more samples to distinguish)
3. Scaling curve has a kink (unexpected non-monotonicity → investigate)

### Compute Budget Allocation

| Phase | Budget | Typical Runs | $/run avg |
|-------|--------|-------------|-----------|
| Pretrain | $300 | 10-15 | $20-30 |
| SFT | $150 | 10-16 | $10-15 |
| RL | $200 | 8-12 | $15-25 |
| Eval/viz | $50 | many | $1-2 |
| **Total** | **$700** | **30-45** | — |

Reserve 20% of each phase's budget for "follow-up" runs that address
surprising results. Don't spend the full budget on the initial plan.
