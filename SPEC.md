# math-nano: Full Experiment Specification

## Mission

Study the scaling of math reasoning capability at sub-1B parameter scale.
Find the smallest model that can achieve meaningful math performance through
the right combination of data mixture, training recipe, and post-training.

Fork: github.com/karpathy/nanochat
Compute: Modal (H100/A100/A10G) + local (MacBook, small models)
Tracking: Weights & Biases project "math-nano"
Package manager: uv

---

## Spec Index

| Spec | File | Purpose |
|------|------|---------|
| Models | `specs/01_models.md` | Model size grid, architecture constraints |
| Pretraining Data | `specs/02_pretrain_data.md` | Data sources, mixtures, tokenization |
| Pretraining Experiments | `specs/03_pretrain_experiments.md` | Pretrain sweep design |
| SFT Data | `specs/04_sft_data.md` | Reasoning trace sources, formatting |
| SFT Experiments | `specs/05_sft_experiments.md` | SFT recipe sweep design |
| Post-Training (RL) | `specs/06_post_training.md` | GRPO, curriculum, reward design |
| Eval Harness | `specs/07_eval_harness.md` | Blessed eval suite, pass@k, variance |
| Infrastructure | `specs/08_infrastructure.md` | Modal, checkpointing, W&B, costs |
| Inference | `specs/09_inference.md` | Serving, sampling, batch eval |
| Results Framework | `specs/10_results_framework.md` | Data collection, analysis, plots |
| Guardrails | `specs/11_guardrails.md` | Validation gates, anti-hacking, audit |
| Agent Pipeline | `specs/12_agent_pipeline.md` | Automated experiment orchestration |
| Stage Validation | `specs/13_stage_validation.md` | Smoke test → sanity → E2E per stage |
| Local Dev & uv | `specs/14_local_dev.md` | Local training, uv setup, laptop runs |
| Bookkeeping | `specs/15_bookkeeping.md` | Data & model registry, provenance, lineage |
| Visualization | `specs/16_visualization.md` | Heatmaps, scaling curves, dashboards |
| Search Strategy | `specs/17_search_strategy.md` | Smart search: eliminate, binary search, budget |
| Harness | `specs/18_harness.md` | Single codepath for all experiments |
| Tests | `specs/19_tests.md` | Unit/integration/smoke tests, coverage |

---

## Experiment Phases (Execution Order)

```
Phase 0: Scaffold       Build repo, data pipelines, eval harness, Modal jobs
Phase 1: Pretrain sweep  5 model sizes x 3 data mixtures = 15 runs
Phase 2: SFT sweep       Top 3 pretrained models x 4 SFT recipes = 12 runs
Phase 3: RL sweep         Top SFT checkpoints x 3 curriculum strategies = 9+ runs
Phase 4: Analysis         Compile scaling curves, write up findings
```

Each phase has explicit entry/exit criteria (see `specs/11_guardrails.md`).
No phase advances until the previous phase's exit criteria are met.
