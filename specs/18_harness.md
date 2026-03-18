# Spec 18: Experiment Harness

## Problem

Without a harness, every experiment is a bespoke script invocation.
Different experiments might:
- Log to W&B differently (or forget to log)
- Skip validation checks
- Save checkpoints in inconsistent formats
- Not register in the model registry
- Not run post-training evals

The harness is the **single codepath** that ALL experiments go through.
It enforces the contract: if it ran through the harness, it's properly tracked.

---

## Architecture

```
launch.py (entry point)
    │
    └→ harness/
        ├── __init__.py
        ├── experiment.py      # Experiment definition & config
        ├── runner.py          # Orchestrates: validate → train → eval → register
        ├── config.py          # Config loading, validation, defaults
        ├── bookkeeper.py      # Data & model registry operations
        ├── metrics.py         # Metric logging contract
        ├── gates.py           # Validation gate checks
        └── search.py          # Search strategy logic
```

Everything calls through the harness. Direct `python base_train.py` is
for local debugging ONLY. Production runs ALWAYS go through `launch.py`.

---

## Experiment Definition

Every experiment is a config object:

```python
@dataclass
class ExperimentConfig:
    # Identity
    experiment_id: str              # e.g., "sft-m-concise"
    stage: Literal["pretrain", "sft", "grpo"]
    phase: str                      # e.g., "2a"

    # Model
    depth: int
    device: str = "auto"            # cuda, mps, cpu, auto

    # Data (pretrain)
    mixture: Optional[str] = None   # e.g., "mix-math-broad"
    token_multiplier: int = 50

    # Data (SFT)
    sft_recipe: Optional[str] = None
    sft_epochs: int = 3
    sft_lr: float = 2e-5
    sft_max_seq_len: int = 2048

    # Data (GRPO)
    rl_curriculum: Optional[str] = None
    rl_kl_coeff: float = 0.05
    rl_group_size: int = 8

    # Parent
    parent_checkpoint: Optional[str] = None  # path or model_id

    # Infra
    gpu: str = "H100"
    timeout_hours: int = 8
    wandb_mode: str = "online"      # online, offline, disabled

    # Eval
    eval_suite: str = "small"       # small, full
    eval_during_training: bool = True
    eval_every: int = 1000          # steps

    # Meta
    tags: list[str] = field(default_factory=list)
    notes: str = ""
```

### Config Files

Stored as YAML in `configs/experiments/`:

```yaml
# configs/experiments/sft-m-concise.yaml
experiment_id: sft-m-concise
stage: sft
phase: "2a"
depth: 16
parent_checkpoint: pt-m-broad-final
sft_recipe: sft-concise-cot
sft_epochs: 3
sft_lr: 2e-5
sft_max_seq_len: 2048
eval_suite: full
eval_every: 200
tags: ["phase-2a", "recipe-comparison"]
notes: "Testing concise CoT recipe on M model"
```

### Config Validation

```python
def validate_config(config: ExperimentConfig) -> list[str]:
    """Validate config before running. Returns list of errors."""
    errors = []

    # Required fields per stage
    if config.stage == "pretrain" and not config.mixture:
        errors.append("Pretrain requires a data mixture")
    if config.stage == "sft" and not config.sft_recipe:
        errors.append("SFT requires a recipe")
    if config.stage in ("sft", "grpo") and not config.parent_checkpoint:
        errors.append(f"{config.stage} requires a parent checkpoint")

    # Parent exists
    if config.parent_checkpoint:
        if not checkpoint_exists(config.parent_checkpoint):
            errors.append(f"Parent checkpoint not found: {config.parent_checkpoint}")

    # Data exists
    if config.mixture:
        if not mixture_data_exists(config.mixture):
            errors.append(f"Data for mixture {config.mixture} not found")

    # No duplicate experiment_id
    if experiment_already_run(config.experiment_id):
        errors.append(f"Experiment {config.experiment_id} already exists. "
                       "Use a new ID or --force to override")

    # Budget check
    estimated_cost = estimate_cost(config)
    remaining = get_remaining_budget(config.stage)
    if estimated_cost > remaining:
        errors.append(f"Estimated cost ${estimated_cost:.2f} exceeds "
                       f"remaining budget ${remaining:.2f}")

    return errors
```

---

## Runner: The Execution Contract

```python
class ExperimentRunner:
    """Enforces the experiment contract."""

    def run(self, config: ExperimentConfig):
        # ═══ PRE-FLIGHT ═══
        errors = validate_config(config)
        if errors:
            raise ConfigError(errors)

        # Check gate for this stage
        if not self.check_gate(config.stage):
            raise GateError(f"Gate for {config.stage} has not passed")

        # ═══ SETUP ═══
        # Initialize W&B
        run = wandb.init(
            project="math-nano",
            group=config.stage,
            name=config.experiment_id,
            tags=self._build_tags(config),
            config=asdict(config),
        )

        # Log provenance
        self._log_provenance(config)

        # ═══ TRAIN ═══
        try:
            if config.stage == "pretrain":
                result = self._run_pretrain(config)
            elif config.stage == "sft":
                result = self._run_sft(config)
            elif config.stage == "grpo":
                result = self._run_grpo(config)
        except Exception as e:
            wandb.log({"error": str(e)})
            self._log_failure(config, e)
            raise

        # ═══ POST-TRAIN EVAL ═══
        eval_results = self._run_eval(config, result.best_checkpoint)

        # ═══ REGISTER ═══
        self._register_model(config, result, eval_results)

        # ═══ LOG COMPLETION ═══
        self._log_completion(config, result, eval_results)
        self._update_experiment_state(config)
        self._append_audit_log(config, "completed")

        wandb.finish()
        return result, eval_results

    def _log_provenance(self, config):
        """Log everything needed to reproduce this run."""
        wandb.config.update({
            "git_hash": get_git_hash(),
            "git_dirty": is_git_dirty(),
            "harness_version": HARNESS_VERSION,
            "data_registry_hash": hash_file("data/registry.json"),
            "eval_manifest_hash": hash_file("data/eval/manifest.json"),
        })

        if config.parent_checkpoint:
            parent = load_model_registry_entry(config.parent_checkpoint)
            wandb.config.update({
                "parent_model_id": config.parent_checkpoint,
                "parent_experiment": parent["experiment_id"],
                "parent_gsm8k": parent["eval_results"].get("gsm8k_pass1_greedy"),
            })
```

---

## Metrics Contract

Every training loop MUST log these metrics through the harness:

### Universal (all stages)

```python
class MetricsContract:
    """Metrics that every training loop must report."""

    # Required every step
    STEP_METRICS = [
        "train/loss",
        "train/lr",
        "train/step",
        "train/tokens_seen",
    ]

    # Required every eval_every steps
    EVAL_METRICS = [
        "eval/gsm8k_pass1",        # from eval harness
    ]

    # Required at end of run
    FINAL_METRICS = [
        "final/train_loss",
        "final/wall_clock_hours",
        "final/tokens_seen",
        "final/cost_usd_estimated",
        "final/checkpoint_path",
    ]

    # Stage-specific additions
    PRETRAIN_METRICS = [
        "val/bpb_math",
        "val/bpb_fineweb",
    ]

    SFT_METRICS = [
        "train/epoch",
        "eval/format_compliance",    # % outputs with \boxed{}
    ]

    GRPO_METRICS = [
        "rl/reward_mean",
        "rl/reward_std",
        "rl/kl_divergence",
        "rl/avg_output_length",
        "rl/curriculum_stage",
    ]
```

### Enforcement

```python
def verify_metrics_logged(experiment_id: str, stage: str) -> list[str]:
    """After a run, verify all required metrics were logged."""
    run = get_wandb_run(experiment_id)
    logged_keys = set(run.summary.keys())

    required = set(MetricsContract.STEP_METRICS +
                   MetricsContract.EVAL_METRICS +
                   MetricsContract.FINAL_METRICS)

    if stage == "pretrain":
        required |= set(MetricsContract.PRETRAIN_METRICS)
    elif stage == "sft":
        required |= set(MetricsContract.SFT_METRICS)
    elif stage == "grpo":
        required |= set(MetricsContract.GRPO_METRICS)

    missing = required - logged_keys
    return list(missing)
```

---

## Harness CLI

```bash
# ═══ Run experiments ═══

# From config file
uv run python launch.py run --config configs/experiments/sft-m-concise.yaml

# From command line (config generated on the fly)
uv run python launch.py run \
  --stage sft --depth 16 \
  --parent pt-m-broad-final \
  --sft-recipe sft-concise-cot

# Batch run
uv run python launch.py batch \
  --configs configs/experiments/sft-m-*.yaml

# ═══ Validation ═══

# Run validation gates
uv run python launch.py gate --check pretrain_to_sft

# Smoke test a config
uv run python launch.py smoke-test --config configs/experiments/sft-m-concise.yaml

# ═══ Registry ═══

# List all registered models
uv run python launch.py registry list

# Show lineage for a model
uv run python launch.py registry lineage sft-m-concise-best

# Compare two models
uv run python launch.py registry compare sft-m-concise-best sft-m-distill-r1-best

# Check for invalidations
uv run python launch.py registry check

# ═══ Eval ═══

# Run blessed eval on a model
uv run python launch.py eval --model sft-m-concise-best --suite full

# ═══ Viz ═══

# Generate all heatmaps
uv run python launch.py viz --all

# Generate dashboard
uv run python launch.py viz --dashboard

# ═══ Status ═══

# What's been done? What's next?
uv run python launch.py status

# Output:
# Phase: 2 (SFT)
# Wave: 3
# Completed: 8/12 planned runs
# Running: 2
# Budget: $112 / $150 spent
# Next gate: sft_to_rl (not yet checked)
# Proposed next: [see agent suggestions]
```

---

## Harness Guarantees

If an experiment ran through the harness, you can be certain that:

| Guarantee | How |
|-----------|-----|
| Config was valid | `validate_config()` ran before training |
| Gate was passed | Gate check ran before training |
| W&B run exists | `wandb.init()` called with full config |
| Provenance logged | git hash, data hashes, parent model recorded |
| All metrics logged | `verify_metrics_logged()` runs post-training |
| Eval was run | Post-training eval is automatic, not optional |
| Model registered | Model added to registry with full metadata |
| Audit trail exists | Action logged to `results/audit_log.jsonl` |
| Budget tracked | Cost estimated and checked against limits |

If ANY of these fail, the experiment is flagged as incomplete/invalid.

---

## Escape Hatch

For debugging and development, you CAN bypass the harness:

```bash
# Direct training (no harness)
uv run python base_train.py --depth 10 --device mps --max-steps 100

# Direct eval (no harness)
uv run python scripts/eval/run_eval.py --checkpoint test.pt --suite small
```

But results from these runs are NOT registered, NOT tracked,
and NOT part of the official experiment record.

The `.wandb_mode=disabled` or `--wandb-mode offline` flags
make this explicit: "this is a dev run, not an experiment."

---

## Extending the Harness

To add a new training stage or experiment type:

1. Add stage to `ExperimentConfig`
2. Add `_run_{stage}()` method to `ExperimentRunner`
3. Add stage-specific metrics to `MetricsContract`
4. Add gate check in `gates.py`
5. Add config validation rules
6. Update `launch.py` CLI

The harness is designed to be extended, not forked. All experiments
should go through the same runner, just with different configs.
