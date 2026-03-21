"""Sample math reasoning datasets to check quality and format."""
import sys
sys.path.insert(0, "/root/math-nano")

DATASETS = {
    "openr1_math": {
        "hf_id": "open-r1/OpenR1-Math-220k",
        "split": "train",
        "problem_field": "problem",
        "solution_field": "solution",
    },
    "mixture_of_thoughts": {
        "hf_id": "open-r1/Mixture-of-Thoughts",
        "subset": "math",
        "split": "train",
        "problem_field": "problem",
        "solution_field": "solution",
    },
    "openmath_reasoning": {
        "hf_id": "nvidia/OpenMathReasoning",
        "split": "cot",
        "problem_field": "problem",
        "solution_field": "solution",
    },
    "qwq_longcot": {
        "hf_id": "amphora/QwQ-LongCoT-130K",
        "split": "train",
        "problem_field": "problem",
        "solution_field": "qwq",
    },
    "am_deepseek_r1": {
        "hf_id": "a-m-team/AM-DeepSeek-R1-Distilled-1.4M",
        "subset": "am_0.9M",
        "split": "train",
    },
}


def check_dataset(name, config):
    from datasets import load_dataset
    print(f"\n{'='*70}")
    print(f"DATASET: {name} ({config['hf_id']})")
    print('='*70)
    try:
        ds = load_dataset(
            config["hf_id"],
            name=config.get("subset"),
            split=config["split"],
            streaming=True,
        )
        samples = list(ds.take(3))
        print(f"Columns: {list(samples[0].keys())}")
        prob_field = config.get("problem_field")
        sol_field = config.get("solution_field")

        for i, s in enumerate(samples):
            problem = s.get(prob_field, "") if prob_field else ""
            solution = s.get(sol_field, "") if sol_field else ""

            if not problem or not solution:
                for k in ["problem", "question", "prompt", "instruction"]:
                    if s.get(k):
                        problem = s[k]; break
                for k in ["solution", "answer", "response", "output", "qwq"]:
                    if s.get(k):
                        solution = s[k]; break
                if not problem and "messages" in s:
                    for m in s["messages"]:
                        if m.get("role") == "user": problem = m.get("content", "")
                        elif m.get("role") == "assistant": solution = m.get("content", "")

            total_chars = len(str(problem)) + len(str(solution))
            has_boxed = "\\boxed{" in str(solution)
            has_think_qwen = "<think>" in str(solution)
            has_think_r1 = "<|begin_of_thought|>" in str(solution)
            think_tag = "<think>" if has_think_qwen else ("<|begin_of_thought|>" if has_think_r1 else "none")
            is_coding = any(x in str(problem).lower() for x in ["python", "function", "stdin", "executable", "def "])

            print(f"\n--- Sample {i} ---")
            print(f"  chars={total_chars}, has_boxed={has_boxed}, think_tag={think_tag}, is_coding={is_coding}")
            print(f"  PROBLEM: {str(problem)[:200].strip()}...")
            print(f"  SOLUTION start: {str(solution)[:300].strip()}...")
            print(f"  SOLUTION end:   ...{str(solution)[-150:].strip()}")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()


for name, config in DATASETS.items():
    check_dataset(name, config)

print("\n\nDone.")
