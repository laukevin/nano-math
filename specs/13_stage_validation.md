# Spec 13: Stage Validation Protocol

## Principle

Every training stage has a 3-tier validation process before real experiments run.
This catches bugs early and cheap, not late and expensive.

```
Tier 1: Smoke Test     → "Does it even run?"       (seconds, CPU)
Tier 2: Sanity Check   → "Does training work?"     (minutes, GPU)
Tier 3: E2E Validation → "Can we reproduce a known result?" (hours, GPU)
```

**No real experiment launches until all 3 tiers pass for that stage.**

---

## PRETRAIN Stage Validation

### Tier 1: Smoke Test (CPU, <60 seconds)

```bash
python launch.py smoke-test --stage pretrain --depth 10
```

What it checks:
| Check | How | Pass Condition |
|-------|-----|----------------|
| Data loads | Load 1 batch from each data source | No errors, shapes correct |
| Model initializes | Create model with depth=10 | No errors, param count matches expected |
| Forward pass works | Run 1 forward pass | Loss is a finite number |
| Backward pass works | Run 1 backward pass | Gradients exist, no NaN |
| Checkpoint save/load | Save checkpoint, reload, compare | State dicts match |
| Eval harness runs | Run eval on 5 GSM8K problems (random answers OK) | No crash |
| W&B mock log | Log 1 dummy metric (dry run, no actual upload) | No error |
| Data mixture loads | Load 1 batch with multi-source weights | Correct proportions (approx) |

Implementation:
```python
def smoke_test_pretrain():
    # 1. Data
    for source in ["fineweb-edu", "openwebmath", "openmathreasoning"]:
        batch = load_batch(source, batch_size=4, seq_len=512)
        assert batch.shape == (4, 512), f"Bad shape: {batch.shape}"
        assert batch.dtype == torch.long
        assert batch.max() < 50257, "Token ID out of range"

    # 2. Model
    model = create_model(depth=10)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    # 3. Forward
    loss = model(batch).loss
    assert torch.isfinite(loss), f"Loss not finite: {loss}"

    # 4. Backward
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN grad in {name}"

    # 5. Checkpoint roundtrip
    save_checkpoint(model, "/tmp/smoke_test.pt")
    model2 = load_checkpoint("/tmp/smoke_test.pt", depth=10)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"Mismatch in {n1}"

    # 6. Eval
    result = run_eval(model, "gsm8k", mode="quick-subset", n=5, device="cpu")
    # Don't check accuracy — untrained model will score ~0. Just check it runs.

    print("SMOKE TEST PASSED")
```

### Tier 2: Sanity Check (GPU, ~5 minutes)

```bash
python launch.py sanity --stage pretrain --depth 10
```

What it checks:
| Check | How | Pass Condition |
|-------|-----|----------------|
| Loss decreases | Train 200 steps | Final loss < initial loss by >10% |
| Loss curve shape | Plot loss over 200 steps | Monotonically decreasing (mostly) |
| LR schedule works | Check LR at step 0, 100, 200 | Matches expected schedule |
| Throughput reasonable | Measure tok/s | Within 50% of expected for GPU type |
| Val BPB computable | Run val every 50 steps | Finite values, not constant |
| W&B logs correctly | Upload to a "sanity-check" W&B group | Metrics visible in dashboard |
| Gradient norm stable | Log grad norm | No explosion (stays < 10) |
| Memory usage okay | Check GPU memory | Peak < 80% of GPU RAM |

Implementation:
```python
def sanity_check_pretrain():
    model = create_model(depth=10)
    optimizer = create_optimizer(model)

    initial_loss = None
    losses = []

    for step in range(200):
        batch = next(train_dataloader)
        loss = train_step(model, batch, optimizer)
        losses.append(loss)

        if step == 0:
            initial_loss = loss
            print(f"Initial loss: {loss:.4f}")

        if step % 50 == 0:
            val_bpb = compute_val_bpb(model, val_dataloader)
            print(f"Step {step}: loss={loss:.4f}, val_bpb={val_bpb:.4f}")
            wandb.log({"step": step, "loss": loss, "val_bpb": val_bpb})

    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss

    assert improvement > 0.10, (
        f"Loss didn't decrease enough: {initial_loss:.4f} → {final_loss:.4f} "
        f"({improvement:.1%} improvement, need >10%)"
    )

    # Check for loss spikes
    max_loss = max(losses[10:])  # skip first 10 (warmup noise)
    assert max_loss < initial_loss * 2, "Loss spiked during training"

    # Check throughput
    elapsed = timer.elapsed()
    tokens_processed = 200 * batch_size * seq_len
    tok_per_sec = tokens_processed / elapsed
    print(f"Throughput: {tok_per_sec:,.0f} tok/s")

    print("SANITY CHECK PASSED")
```

### Tier 3: E2E Validation (GPU, ~1-2 hours)

```bash
python launch.py validate --stage pretrain --depth 10 --mixture mix-math-broad
```

What it checks:
| Check | How | Pass Condition |
|-------|-----|----------------|
| Standard loss curve | Train 2000 steps | Smooth decrease, matches expected shape |
| Val BPB converges | Compute every 200 steps | Clearly decreasing trend |
| Data mixture works | Log source distribution | Matches configured weights (±5%) |
| Checkpoint resume | Save at 1000, resume at 1000, train to 2000 | Final loss matches continuous run (±1%) |
| Full eval pipeline | Run GSM8K eval at end | Runs successfully (score can be ~0) |
| Results JSON saved | Check output file | Valid JSON with all expected fields |
| Reproducibility | Run same config twice with same seed | Losses match within 1% |

**This is the "known result" tier.** For pretrain, "known result" = loss curve
that follows the expected shape. We don't expect great GSM8K performance from
2000 steps of pretraining, but we expect:
- Loss at step 2000 should be substantially lower than step 0
- Val BPB should track train loss (no big train/val gap at 2000 steps)
- No crashes, no NaN, no OOM

```python
def e2e_validate_pretrain():
    # Full run: 2000 steps with all the trimmings
    run_training(
        depth=10,
        mixture="mix-math-broad",
        max_steps=2000,
        eval_every=500,
        checkpoint_every=1000,
        wandb_group="e2e-validation",
    )

    # Verify loss curve
    losses = load_wandb_metric("train/loss")
    assert losses[-1] < losses[0] * 0.5, "Loss didn't halve in 2000 steps"

    # Verify val_bpb trend
    val_bpbs = load_wandb_metric("val/bpb_math")
    assert all(val_bpbs[i] >= val_bpbs[i+1] - 0.05
               for i in range(len(val_bpbs)-1)), "Val BPB not decreasing"

    # Verify checkpoint resume
    # (Already trained 2000 steps continuously)
    # Now train 1000 steps, save, resume, train 1000 more
    run_training(depth=10, mixture="mix-math-broad", max_steps=1000,
                 checkpoint_dir="/tmp/resume_test/")
    run_training(depth=10, mixture="mix-math-broad", max_steps=2000,
                 resume_from="/tmp/resume_test/step_001000.pt")
    resumed_final_loss = load_final_metric("train/loss")
    assert abs(losses[-1] - resumed_final_loss) / losses[-1] < 0.02, \
        "Resume produces different result"

    # Verify eval output
    eval_result = load_json("results/eval/e2e-validation_gsm8k.json")
    assert "pass_at_1_greedy" in eval_result["results"]["gsm8k"]

    print("E2E VALIDATION PASSED")
```

---

## SFT Stage Validation

### Tier 1: Smoke Test (CPU, <60 seconds)

```bash
python launch.py smoke-test --stage sft --depth 10
```

| Check | How | Pass Condition |
|-------|-----|----------------|
| SFT data loads | Load 1 batch from SFT dataset | Chat format correct, shapes right |
| Pretrained checkpoint loads | Load a pretrain checkpoint | No errors |
| SFT forward pass | Forward pass on SFT formatted batch | Loss is finite |
| SFT loss masking | Verify loss only on assistant tokens | System/user tokens have zero loss |
| Chat template works | Format 5 samples, detokenize, inspect | Looks correct |
| Sequence length handling | Load samples near max_seq_len | Proper truncation, no crash |

Key SFT-specific check — **loss masking**:
```python
def verify_loss_masking():
    """SFT should only compute loss on assistant (response) tokens,
    not on the system prompt or user message."""
    batch = create_sft_batch(samples=[{
        "system": "You are a math assistant.",
        "user": "What is 2+2?",
        "assistant": "2+2 = 4. The answer is \\boxed{4}"
    }])

    # Check that loss mask has zeros for system+user, ones for assistant
    loss_mask = batch["loss_mask"]
    assistant_start = find_assistant_start(batch)
    assert loss_mask[:assistant_start].sum() == 0, "Loss on non-assistant tokens!"
    assert loss_mask[assistant_start:].sum() > 0, "No loss on assistant tokens!"
```

### Tier 2: Sanity Check (GPU, ~10 minutes)

```bash
python launch.py sanity --stage sft --depth 10 --recipe sft-concise-cot
```

| Check | How | Pass Condition |
|-------|-----|----------------|
| SFT loss decreases | Train 200 steps | Loss drops by >20% |
| No loss explosion | Monitor loss | Never exceeds 2x initial |
| Format compliance improves | Check \boxed{} rate at step 0 vs 200 | Increases |
| GSM8K eval runs | Quick eval at step 200 | Runs without crash |
| LR warmup works | Check LR values | Matches cosine schedule |
| Overfitting check | Compare train loss and eval loss | Gap is < 2x |

**Key SFT sanity signal:** After 200 steps, the model should start producing
outputs that look like math solutions, even if wrong. Sample 5 outputs and
check they contain numbers and basic structure.

```python
def sanity_check_sft():
    model = load_pretrain_checkpoint(depth=10)

    # Sample outputs BEFORE SFT
    pre_sft_outputs = [generate(model, p) for p in sample_problems[:5]]
    pre_sft_boxed_rate = sum(1 for o in pre_sft_outputs if "\\boxed" in o) / 5

    # Train 200 steps
    train_sft(model, recipe="sft-concise-cot", max_steps=200)

    # Sample outputs AFTER 200 steps
    post_sft_outputs = [generate(model, p) for p in sample_problems[:5]]
    post_sft_boxed_rate = sum(1 for o in post_sft_outputs if "\\boxed" in o) / 5

    print(f"Format compliance: {pre_sft_boxed_rate:.0%} → {post_sft_boxed_rate:.0%}")

    # We expect SOME format learning even in 200 steps
    # (not necessarily a big jump, but model should start trying)
    assert post_sft_boxed_rate >= pre_sft_boxed_rate, \
        "Format compliance didn't improve at all"

    print("SFT SANITY CHECK PASSED")
```

### Tier 3: E2E Validation (GPU, ~1 hour)

```bash
python launch.py validate --stage sft --depth 10 --recipe sft-concise-cot
```

| Check | How | Pass Condition |
|-------|-----|----------------|
| GSM8K improves over pretrain | Compare pre/post SFT eval | Improvement > 0 (any) |
| Loss converges | Train full 3 epochs on small dataset | Loss curve flattens |
| Best checkpoint saved | Check checkpoint directory | best_gsm8k.pt exists |
| No catastrophic forgetting | Check general text generation | Still produces coherent English |
| Epoch dynamics | GSM8K at end of each epoch | Not degrading after epoch 1 |

**Expected known result:** Even a 50M model with 3 epochs of SFT on GSM8K-style
data should get *some* GSM8K problems right (>2%). If it gets literally zero,
something is wrong with the pipeline.

---

## GRPO Stage Validation

### Tier 1: Smoke Test (CPU, <60 seconds)

```bash
python launch.py smoke-test --stage grpo --depth 10
```

| Check | How | Pass Condition |
|-------|-----|----------------|
| SFT checkpoint loads for RL | Load checkpoint + init GRPO | No errors |
| HF conversion works | Convert nanochat → HF format | Model produces same outputs |
| Reward function works | Compute reward on 5 known (output, answer) pairs | Correct rewards |
| Group generation works | Generate 8 completions for 1 prompt | 8 distinct completions |
| GRPO loss computes | 1 GRPO step with dummy rewards | Finite loss, gradients exist |
| Curriculum datasets load | Load GSM8K, AMC, AIME for RL | Correct format |

Key GRPO-specific check — **reward function correctness**:
```python
def verify_reward_function():
    test_cases = [
        ("The answer is \\boxed{42}", "42", 1.0),
        ("The answer is \\boxed{42}", "43", 0.0),
        ("I don't know", "42", 0.0),
        ("#### 42", "42", 1.0),
        ("The answer is 42.", "42", 1.0),
        ("\\boxed{3.14}", "3.14", 1.0),
        ("\\boxed{3.14}", "3.1400", 1.0),  # numeric equivalence
        ("", "42", 0.0),  # empty output
    ]
    for output, truth, expected_reward in test_cases:
        reward = compute_reward(output, truth)
        assert reward == expected_reward, \
            f"Reward({output!r}, {truth!r}) = {reward}, expected {expected_reward}"
```

### Tier 2: Sanity Check (GPU, ~15 minutes)

```bash
python launch.py sanity --stage grpo --depth 10
```

| Check | How | Pass Condition |
|-------|-----|----------------|
| Reward signal exists | 50 GRPO steps | Mean reward > 0 (model solves some GSM8K) |
| Reward increases | Compare step 0 vs step 50 | Positive trend |
| KL stays bounded | Monitor KL divergence | KL < 5 after 50 steps |
| Output length stable | Monitor avg output length | Not collapsing to 0 or exploding |
| No reward hacking | Check reward vs eval | Consistent direction |
| Diversity maintained | Check unique outputs in group | >50% unique |

**Critical sanity check:** If reward_mean is 0.0 for all 50 steps, the model
can't solve ANY problems. This means SFT was insufficient or the reward function
is broken. Stop and investigate — don't waste compute on a full RL run.

```python
def sanity_check_grpo():
    model = load_sft_checkpoint(depth=10)
    grpo_trainer = setup_grpo(model, dataset="gsm8k")

    rewards = []
    for step in range(50):
        metrics = grpo_trainer.step()
        rewards.append(metrics["reward_mean"])

        if step == 10 and max(rewards) == 0.0:
            print("WARNING: Zero reward for first 10 steps. "
                  "Model may not be capable enough for RL.")

    if max(rewards) == 0.0:
        print("SANITY CHECK FAILED: No reward signal after 50 steps. "
              "SFT model cannot solve any GSM8K problems.")
        return False

    # Check reward trend
    first_10_avg = sum(rewards[:10]) / 10
    last_10_avg = sum(rewards[40:]) / 10
    print(f"Reward: {first_10_avg:.3f} → {last_10_avg:.3f}")

    # Check KL
    kl = grpo_trainer.get_kl()
    assert kl < 5.0, f"KL too high: {kl:.2f}"

    print("GRPO SANITY CHECK PASSED")
```

### Tier 3: E2E Validation (GPU, ~2 hours)

```bash
python launch.py validate --stage grpo --depth 10 --curriculum easy2hard
```

| Check | How | Pass Condition |
|-------|-----|----------------|
| GSM8K improves over SFT | Compare pre/post RL eval | Any improvement |
| Curriculum advancement | Does model hit GSM8K >40% gate? | Reaches at least stage 2 |
| Best checkpoint saved | Check for best_gsm8k.pt | Exists and loads |
| No reward hacking | Full hacking detection suite | All checks pass |
| Multi-dataset eval works | Run eval on gsm8k + math500 + amc + aime | All produce results |
| Results JSON complete | Check output | All fields present |

---

## Validation Summary

### Full Validation Sequence

```
═══════════════════════════════════════════════
STAGE: PRETRAIN
═══════════════════════════════════════════════
  Tier 1: Smoke Test (CPU)        □ PASS / FAIL
  Tier 2: Sanity Check (GPU)      □ PASS / FAIL
  Tier 3: E2E Validation (GPU)    □ PASS / FAIL

═══════════════════════════════════════════════
STAGE: SFT
═══════════════════════════════════════════════
  Tier 1: Smoke Test (CPU)        □ PASS / FAIL
  Tier 2: Sanity Check (GPU)      □ PASS / FAIL
  Tier 3: E2E Validation (GPU)    □ PASS / FAIL

═══════════════════════════════════════════════
STAGE: GRPO
═══════════════════════════════════════════════
  Tier 1: Smoke Test (CPU)        □ PASS / FAIL
  Tier 2: Sanity Check (GPU)      □ PASS / FAIL
  Tier 3: E2E Validation (GPU)    □ PASS / FAIL
```

### When to Re-Validate

Re-run validation when:
- Code changes to the training loop
- Data pipeline changes
- New data source added
- nanochat version updated
- Modal image changes

Don't re-validate for:
- New experiment configs (same code, different hyperparameters)
- Adding new eval datasets (eval code is independently validated)

### Validation Costs

| Test | GPU Time | Cost |
|------|----------|------|
| All smoke tests | 0 (CPU) | $0 |
| All sanity checks | ~30 min | ~$2 |
| All E2E validations | ~4 hours | ~$14 |
| **Total validation** | **~4.5 hours** | **~$16** |

This is <3% of the total project budget. Cheap insurance.

### Validation Script

```bash
# Run all validations for all stages
python launch.py validate-all

# Run just one tier for one stage
python launch.py smoke-test --stage pretrain
python launch.py sanity --stage sft
python launch.py validate --stage grpo

# Quick check: just smoke tests for all stages
python launch.py smoke-test --all
```

Output: structured report saved to `results/validation_report.json`
```json
{
  "timestamp": "2026-03-18T10:00:00Z",
  "git_hash": "abc123",
  "results": {
    "pretrain": {
      "smoke_test": {"passed": true, "duration_s": 45, "checks": [...]},
      "sanity_check": {"passed": true, "duration_s": 300, "checks": [...]},
      "e2e_validation": {"passed": true, "duration_s": 3600, "checks": [...]}
    },
    "sft": {...},
    "grpo": {...}
  },
  "overall": "PASSED"
}
```
