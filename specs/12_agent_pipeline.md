# Spec 12: Agent Automation Pipeline

## Goal

Enable automated agents (Claude Code instances) to run experiments in waves,
while keeping humans in the loop for analysis and decision-making.

## Operating Model

```
Human (you) ←→ Agent (Claude Code)
     │                │
     │ Designs         │ Executes
     │ Analyzes        │ Monitors
     │ Decides         │ Reports
     │                │
     └────────────────┘
```

### Division of Labor

| Task | Human | Agent |
|------|-------|-------|
| Design experiment matrix | Primary | Suggests |
| Launch training runs | Approves | Executes |
| Monitor training health | Spot-checks | Continuous |
| Run validation gates | Reviews | Executes |
| Compile results | Reviews | Executes |
| Analyze scaling curves | Primary | Generates plots |
| Decide next experiments | Primary | Proposes |
| Debug failures | Collaborate | Investigate |
| Write summaries | Collaborate | Drafts |

## Agent Capabilities

### What the Agent Can Do Autonomously

```bash
# 1. Launch approved experiments
python launch.py run --depth 12 --experiment pt-s-broad

# 2. Monitor running jobs
python launch.py status

# 3. Run evals on completed checkpoints
python scripts/eval/run_eval.py --checkpoint $CKPT --datasets gsm8k,math500

# 4. Compile results
python scripts/results/compile.py --results-dir results/eval/ --output results/compiled/

# 5. Generate plots
python scripts/results/plot.py --data results/compiled/full_results.csv --plot all

# 6. Run validation gates
python scripts/validate/run_gate.py --gate pretrain_to_sft

# 7. Check W&B for run status and metrics
python scripts/monitor/check_wandb.py --experiment-id pt-s-broad
```

### What Requires Human Approval

1. **Phase transitions** — Agent runs the gate, human reviews results
2. **New experiment designs** — Agent proposes, human approves
3. **Budget increases** — Agent flags limit hit, human authorizes
4. **Code changes** — Agent proposes changes, human reviews diff
5. **Anomaly resolution** — Agent detects issue, human decides action

## Experiment Execution Flow

### Conversation Pattern

Each "wave" of experiments follows this loop:

```
1. Agent proposes experiment batch (what to run, why, cost estimate)
2. Human approves / modifies / rejects
3. Agent launches approved runs
4. Agent monitors runs, reports progress
5. Runs complete → Agent runs validation gates
6. Agent compiles results, generates plots
7. Agent presents findings: "Here's what happened..."
8. Human analyzes, decides next wave
9. Goto 1
```

### Phase 1 Example

```
Wave 1 (Pilot):
  Agent → "Launching pt-s-general, pt-s-broad, pt-s-heavy. ~$42, ~4h each."
  Human → "Go."
  [runs complete]
  Agent → "All 3 done. Gate checks pass. Results:
           mix-general: GSM8K 0.01, math_bpb 3.2
           mix-broad:   GSM8K 0.02, math_bpb 2.8
           mix-heavy:   GSM8K 0.02, math_bpb 2.5
           Broad and heavy both beat general. Recommend running both on XS and M."
  Human → "Run broad on all sizes. Skip mix-general for XS/M/L/XL."

Wave 2 (Size sweep):
  Agent → "Launching pt-xs-broad, pt-m-broad, pt-l-broad, pt-xl-broad. ~$130."
  Human → "Go."
  [runs complete]
  Agent → "Scaling curve generated. [shows plot]
           Math bpb drops log-linearly with params. Saturation point around XL.
           Ready for Phase 2 (SFT). Gate 1 checks pass."
  Human → "Let's start SFT. Use distill-r1 and concise-cot on the M model first."
```

### Phase 2 Example

```
Wave 3 (SFT recipe comparison):
  Agent → "Launching sft-m-distill-r1, sft-m-concise on best M pretrain. ~$14."
  Human → "Go."
  [runs complete]
  Agent → "Results:
           distill-r1: GSM8K 28%, MATH500 8%
           concise:    GSM8K 34%, MATH500 11%
           Concise wins. Shorter traces work better for 130M.
           Recommend running concise on all sizes."
  Human → "Interesting. Also try the quality recipe — I want to see if 30K
           high-quality samples beats 100K medium-quality."
```

## Launch Interface

### `launch.py` — Single Entry Point

```bash
# === VALIDATION (must pass before real training) ===

# Quick smoke test — 10 steps, CPU, checks data loading + model init
python launch.py smoke-test --depth 10

# Sanity run — 100 steps, GPU, checks loss curve + eval pipeline
python launch.py sanity --depth 10 --stage pretrain

# E2E validation — 500 steps, GPU, expects measurable loss decrease
python launch.py validate --depth 10 --stage pretrain --mixture mix-math-broad

# === REAL EXPERIMENTS ===

# Single run
python launch.py run --depth 12 --experiment pt-s-broad

# Batch launch (multiple runs)
python launch.py batch --experiments pt-s-general,pt-s-broad,pt-s-heavy

# Full sweep (all sizes with one config)
python launch.py sweep --mixture mix-math-broad --stage pretrain

# === MONITORING ===

# Check status of all running Modal jobs
python launch.py status

# Check specific experiment
python launch.py status --experiment pt-s-broad

# === EVAL ===

# Eval a checkpoint
python launch.py eval --checkpoint /checkpoints/d12/pretrain/final.pt --datasets gsm8k,math500

# Eval all final checkpoints for a phase
python launch.py eval-all --stage pretrain --mode full

# === RESULTS ===

# Compile all results into CSVs + plots
python launch.py compile

# Generate phase summary
python launch.py summarize --phase 1
```

## Agent Reporting Format

When presenting results, agents should use this structure:

```markdown
## Wave N Results: [description]

### Runs Completed
| Experiment | Status | Duration | Cost |
|------------|--------|----------|------|
| pt-s-broad | ✓ | 4.2h | $14.70 |

### Key Metrics
| Experiment | GSM8K | MATH500 | Math BPB |
|------------|-------|---------|----------|
| pt-s-broad | 0.02 | 0.01 | 2.81 |

### Gate Status
- [✓] Checkpoints saved
- [✓] Loss decreased
- [✓] Model generates text
- [✓] W&B logging complete

### Observations
- [Key finding 1]
- [Key finding 2]

### Recommended Next Steps
1. [Suggestion with rationale]
2. [Alternative option]

### Decision Needed
[What the human needs to decide before next wave]
```

## Error Handling

### Run Failure
```
Agent detects: Modal job failed (OOM, timeout, crash)
Agent action:
  1. Save error logs
  2. Diagnose likely cause (OOM → reduce batch size, timeout → increase limit)
  3. Report to human with proposed fix
  4. Wait for approval before retrying
```

### Anomalous Results
```
Agent detects: Results look wrong (loss went UP, negative accuracy, etc.)
Agent action:
  1. Flag anomaly
  2. Check for obvious causes (wrong checkpoint, data loading issue)
  3. If not obvious: present to human for investigation
  4. DO NOT silently retry or ignore
```

### Budget Overrun
```
Agent detects: Next run would exceed phase budget
Agent action:
  1. Report current spend + remaining budget
  2. Propose: skip remaining runs, or request budget increase
  3. Wait for human decision
```

## State Tracking

Agent maintains experiment state in `results/experiment_state.json`:

```json
{
  "current_phase": 1,
  "current_wave": 2,
  "completed_experiments": ["pt-s-general", "pt-s-broad", "pt-s-heavy"],
  "running_experiments": ["pt-m-broad", "pt-l-broad"],
  "pending_experiments": ["pt-xl-broad"],
  "total_spend_usd": 56.70,
  "phase_budget_remaining_usd": 243.30,
  "last_gate_status": {
    "gate": "preflight",
    "passed": true,
    "timestamp": "2026-03-18T10:00:00Z"
  },
  "decisions_log": [
    {
      "wave": 1,
      "decision": "Drop mix-general, run mix-broad on all sizes",
      "decided_by": "human",
      "timestamp": "2026-03-19T14:00:00Z"
    }
  ]
}
```

This file is the source of truth for what's been done and what's next.
Agents read it before proposing next steps.
Agents update it after completing actions.
