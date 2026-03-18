# Spec 11: Guardrails & Validation

## Purpose

Ensure experiment integrity. Prevent:
1. Reward hacking (RL models gaming metrics without real capability)
2. Data leakage (eval data in training sets)
3. Silent failures (runs that look okay but produced garbage)
4. Wasted compute (runs that should have been stopped early)
5. Agent misuse (automated agents skipping validation or cutting corners)

## Gate System

Every phase transition requires passing a validation gate.
Gates are automated checks run before proceeding to the next phase.

### Gate 0: Pre-Flight (before any training)

```python
def gate_preflight():
    checks = []

    # Data integrity
    checks.append(verify_shard_format("data/openwebmath/"))
    checks.append(verify_shard_format("data/openmathreasoning/"))
    checks.append(verify_no_eval_leakage(
        train_dirs=["data/openwebmath/", "data/openmathreasoning/"],
        eval_datasets=["gsm8k_test", "math500", "aime"],
    ))

    # Model sanity
    for depth in [10, 12, 16, 20, 24]:
        checks.append(verify_model_loads(depth))
        checks.append(verify_forward_pass(depth, batch_size=4))

    # Eval harness
    checks.append(verify_eval_runs("gsm8k", mode="quick-subset", device="cpu"))
    checks.append(verify_answer_extraction_accuracy() > 0.99)

    # W&B connectivity
    checks.append(verify_wandb_logging())

    # Modal setup
    checks.append(verify_modal_volumes_exist())
    checks.append(verify_modal_secrets_exist())

    return all(checks)
```

### Gate 1: Pretrain → SFT

```python
def gate_pretrain_to_sft():
    checks = []

    # Did training actually happen?
    checks.append(verify_checkpoints_exist("pretrain"))
    checks.append(verify_tokens_seen() > min_expected_tokens)

    # Did loss go down?
    checks.append(verify_loss_decreased("val/bpb_math", threshold=0.1))

    # Is the model not degenerate?
    checks.append(verify_model_generates_text())  # not all padding/EOS
    checks.append(verify_perplexity_reasonable())  # not infinite or zero

    # Were all runs logged to W&B?
    checks.append(verify_wandb_runs_complete("pretrain"))

    # Scaling curve sanity: larger models should have lower loss
    checks.append(verify_scaling_monotonic("val/bpb_math"))

    return all(checks)
```

### Gate 2: SFT → RL

```python
def gate_sft_to_rl():
    checks = []

    # SFT improved GSM8K over pretrain-only?
    checks.append(verify_metric_improved("gsm8k_pass1", "pretrain", "sft"))

    # Model produces formatted answers?
    checks.append(verify_format_compliance() > 0.80)  # >80% use \boxed{}

    # No catastrophic forgetting?
    checks.append(verify_no_catastrophic_forgetting())

    # Best checkpoint was saved?
    checks.append(verify_best_checkpoint_exists("sft"))

    return all(checks)
```

### Gate 3: RL Completion

```python
def gate_rl_complete():
    checks = []

    # RL didn't make things worse?
    checks.append(verify_metric_not_degraded("gsm8k_pass1", "sft", "grpo"))

    # No reward hacking? (see below)
    checks.append(verify_no_reward_hacking())

    # Final evals ran on all datasets?
    checks.append(verify_final_evals_complete())

    return all(checks)
```

## Reward Hacking Detection

### What to Watch For

| Hacking Type | Signal | Detection |
|-------------|--------|-----------|
| Empty output | High reward but empty/short outputs | `avg_output_length < 20` |
| Repetition | Model repeats answer pattern to fill space | Unique n-gram ratio < 0.3 |
| Format gaming | \boxed{} with random numbers | `reward > 0.5 but eval_gsm8k < 0.1` |
| Reward-eval divergence | Training reward up, eval metrics flat | Correlation < 0.5 |
| Exploitation of extraction | Model outputs something that tricks extractor | Manual spot-check |
| KL collapse | Policy became degenerate | `kl_divergence > 15` |

### Automated Detection

```python
def verify_no_reward_hacking():
    checks = []

    # 1. Reward-eval consistency
    # If reward says >50% correct, GSM8K eval should be >30%
    reward_mean = get_metric("rl/reward_mean", last_100_steps=True)
    gsm8k_eval = get_metric("eval/gsm8k_pass1", latest=True)
    if reward_mean > 0.5 and gsm8k_eval < 0.15:
        checks.append(False)  # SUSPICIOUS
        log_warning("Reward-eval divergence detected!")

    # 2. Output quality
    avg_len = get_metric("rl/avg_output_length", last_100_steps=True)
    if avg_len < 20:
        checks.append(False)  # Degenerate short outputs

    # 3. Diversity
    recent_outputs = get_recent_generations(n=100)
    unique_ratio = len(set(recent_outputs)) / len(recent_outputs)
    if unique_ratio < 0.3:
        checks.append(False)  # Too many identical outputs

    # 4. KL divergence
    kl = get_metric("rl/kl_divergence", latest=True)
    if kl > 15:
        checks.append(False)  # Policy collapsed

    return all(checks)
```

### Manual Spot-Check Protocol

At the end of every RL run, save 50 random (prompt, completion, reward) triples.
Human reviews these for:
- Does the reasoning make sense?
- Is the answer actually correct (not just matching extraction)?
- Is the output fluent and coherent?
- Any signs of degenerate behavior?

Save spot-check results as an artifact:
```json
{
  "experiment_id": "grpo-m-easy2hard",
  "spot_check_date": "2026-03-25",
  "samples_reviewed": 50,
  "issues_found": [
    {"sample_idx": 12, "issue": "correct answer but nonsensical reasoning"},
    {"sample_idx": 37, "issue": "repetitive filler text before answer"}
  ],
  "overall_quality": "acceptable",  // or "concerning" or "failed"
  "reviewer": "kevin"
}
```

## Data Leakage Prevention

### Pre-Training Dedup

Before any training begins:

```python
def check_eval_leakage():
    """Verify no eval problems appear in training data."""
    eval_problems = load_all_eval_problems()  # GSM8K test, MATH500, AIME

    for source in ["openwebmath", "openmathreasoning", "fineweb-edu"]:
        train_text = load_tokenized_text(f"data/{source}/")
        for problem in eval_problems:
            # Exact substring match
            if problem.text in train_text:
                raise LeakageError(f"Eval problem found in {source}!")
            # Fuzzy match (catch paraphrases)
            if fuzzy_match(problem.text, train_text, threshold=0.9):
                log_warning(f"Possible paraphrase in {source}: {problem.id}")

    print("No leakage detected.")
```

### SFT Data Dedup

Same check for SFT datasets. Especially important for MetaMath
(which is derived from GSM8K train — ensure test problems aren't included).

## Run Health Monitoring

### During Training (every 1000 steps)

```python
def health_check(step, metrics):
    """Run during training. Returns warnings/stops."""

    # Loss sanity
    if metrics["train/loss"] > 100 or math.isnan(metrics["train/loss"]):
        STOP("Loss exploded or NaN")

    # Loss not decreasing (after warmup)
    if step > 5000 and metrics["train/loss"] > initial_loss * 0.95:
        WARN("Loss hasn't decreased significantly after 5000 steps")

    # Throughput check
    if metrics["meta/tokens_per_second"] < expected_throughput * 0.5:
        WARN("Throughput is less than half expected — possible data loading bottleneck")

    # Gradient norm (if logged)
    if "train/grad_norm" in metrics and metrics["train/grad_norm"] > 100:
        WARN("Gradient norm very high — potential instability")
```

### After Training (per-run validation)

```python
def post_run_validation(experiment_id):
    """Run after training completes, before declaring success."""

    # 1. Checkpoint loads successfully
    model = load_checkpoint(get_best_checkpoint(experiment_id))

    # 2. Model generates reasonable text
    output = generate(model, "What is 2 + 2?", max_tokens=100)
    assert len(output) > 5, "Model generated too-short output"
    assert "4" in output, "Model can't do 2+2"  # basic sanity

    # 3. Eval produces non-zero results (for SFT/RL stages)
    if stage in ["sft", "grpo"]:
        eval_result = run_eval(model, "gsm8k", mode="quick-subset")
        assert eval_result["pass_at_1"] > 0, "Zero GSM8K score after SFT/RL"

    # 4. W&B run has all required metrics
    verify_wandb_metrics_complete(experiment_id)

    # 5. Results JSON was saved
    assert os.path.exists(f"results/eval/{experiment_id}_gsm8k.json")
```

## Agent Automation Guardrails

When agents (automated Claude Code instances) run experiments:

### What Agents CAN Do
- Launch training jobs via `launch.py`
- Monitor W&B dashboards
- Run eval on completed checkpoints
- Compile results and generate plots
- Suggest next experiments based on results

### What Agents CANNOT Do Without Human Approval
- Skip any validation gate
- Mark a gate as "passed" when checks fail
- Delete checkpoints or results
- Modify the eval harness code
- Change reward functions during RL
- Increase budget beyond authorized amount
- Start Phase N+1 before Phase N gates pass

### Agent Audit Trail

Every agent action is logged:
```json
{
  "timestamp": "2026-03-20T14:00:00Z",
  "agent_id": "agent-001",
  "action": "launch_experiment",
  "experiment_id": "pt-s-broad",
  "params": {"depth": 12, "mixture": "mix-math-broad", "token_mult": 50},
  "gate_status": "gate_0_passed",
  "human_approved": true
}
```

Saved to `results/audit_log.jsonl`. Append-only, never deleted.

### Budget Guardrail

```python
MAX_BUDGET_PER_PHASE = {
    "pretrain": 500,   # USD
    "sft": 200,
    "grpo": 300,
}

def check_budget(phase, new_job_estimated_cost):
    spent = sum_costs_for_phase(phase)
    if spent + new_job_estimated_cost > MAX_BUDGET_PER_PHASE[phase]:
        raise BudgetExceeded(
            f"Phase {phase}: spent ${spent:.2f}, "
            f"new job ~${new_job_estimated_cost:.2f}, "
            f"limit ${MAX_BUDGET_PER_PHASE[phase]}"
        )
```

## Validation Script

Single entry point for all gates:

```bash
# Run specific gate
python scripts/validate/run_gate.py --gate preflight
python scripts/validate/run_gate.py --gate pretrain_to_sft
python scripts/validate/run_gate.py --gate sft_to_rl
python scripts/validate/run_gate.py --gate rl_complete

# Run all checks for current state
python scripts/validate/run_gate.py --auto
```

Output: pass/fail with details for each check.
If any check fails, print clear instructions on what to investigate.
