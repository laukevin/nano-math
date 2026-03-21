"""Compute char-length histogram for mixture_of_thoughts math subset."""
import sys
sys.path.insert(0, "/root/math-nano")

N_SAMPLES = 2000

def get_problem_solution(s):
    problem = solution = ""
    for m in s.get("messages", []):
        if m.get("role") == "user":
            problem = m.get("content", "")
        elif m.get("role") == "assistant":
            solution = m.get("content", "")
    return problem, solution

from datasets import load_dataset
print("Streaming mixture_of_thoughts/math...")
ds = load_dataset("open-r1/Mixture-of-Thoughts", name="math", split="train", streaming=True)

lengths = []
for i, s in enumerate(ds):
    if i >= N_SAMPLES:
        break
    prob, sol = get_problem_solution(s)
    lengths.append(len(prob) + len(sol))

lengths.sort()
total = len(lengths)

import statistics
print(f"\nN={total} samples")
print(f"min={lengths[0]:,}  max={lengths[-1]:,}")
print(f"mean={statistics.mean(lengths):,.0f}  median={statistics.median(lengths):,.0f}")
print(f"p50={lengths[int(total*0.50)]:,}")
print(f"p75={lengths[int(total*0.75)]:,}")
print(f"p90={lengths[int(total*0.90)]:,}")
print(f"p95={lengths[int(total*0.95)]:,}")
print(f"p99={lengths[int(total*0.99)]:,}")

# Text histogram with 20 buckets
print("\n--- Char-length histogram (2000 samples) ---")
import math
bucket_size = 5000
max_len = lengths[-1]
buckets = {}
for l in lengths:
    b = (l // bucket_size) * bucket_size
    buckets[b] = buckets.get(b, 0) + 1

bar_width = 40
max_count = max(buckets.values())
print(f"{'Chars':>12}  {'Count':>5}  {'%':>5}  Bar")
print("-" * 65)
for b in sorted(buckets):
    count = buckets[b]
    pct = count / total * 100
    bar = "█" * int(count / max_count * bar_width)
    print(f"{b:>7}-{b+bucket_size-1:<7}  {count:>5}  {pct:>4.1f}%  {bar}")

# Cumulative at key cutoffs
print("\n--- Cumulative % below cutoff ---")
cutoffs = [5000, 7000, 10000, 14000, 20000, 30000, 50000]
for c in cutoffs:
    below = sum(1 for l in lengths if l < c)
    print(f"  < {c:>6,} chars: {below/total*100:>5.1f}%  ({below}/{total})")

# Suggest curriculum splits (target ~equal samples per phase, seq-len aware)
print("\n--- Suggested curriculum splits ---")
print("  seq2048 -> max_chars ~7000  (chars = seq_len * 3.5)")
print("  seq4096 -> max_chars ~14000")
print("  seq8192 -> max_chars ~28000 (if needed)")
