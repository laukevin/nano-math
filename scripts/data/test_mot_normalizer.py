"""Test normalize_mixture_of_thoughts on 100 samples."""
import sys, json
sys.path.insert(0, "/root/math-nano")

from scripts.data.normalize_dataset import normalize_mixture_of_thoughts

from datasets import load_dataset
ds = load_dataset("open-r1/Mixture-of-Thoughts", name="math", split="train", streaming=True)

ok = skipped = 0
has_boxed = has_think_leak = 0

for i, row in enumerate(ds):
    if i >= 100:
        break
    result = normalize_mixture_of_thoughts(row)
    if result is None:
        skipped += 1
        continue
    sol = result["solution"]
    if "\\boxed{" in sol:
        has_boxed += 1
    if "<think>" in sol or "</think>" in sol or "<|begin_of_thought|>" in sol:
        has_think_leak += 1
        print(f"  [LEAK] Sample {i} still has think tags!")
    ok += 1
    if i < 3:
        print(f"\n=== Sample {i} ===")
        print(f"  problem ({len(result['problem'])} chars): {result['problem'][:120]}...")
        print(f"  solution ({len(result['solution'])} chars):")
        print(f"    START: {sol[:200]}...")
        print(f"    END:   ...{sol[-150:]}")
        print(f"  has_boxed={('\\\\boxed{' in sol)}, think_leaked={('<think>' in sol)}")

print(f"\n--- Results for 100 samples ---")
print(f"  ok={ok}, skipped={skipped}")
print(f"  has_boxed: {has_boxed}/{ok} ({has_boxed/ok*100:.0f}%)")
print(f"  think_tag_leaked: {has_think_leak}/{ok}")
