"""Analyze a math dataset for quality, format, and content properties.

Pipeline:
  Phase 1 (50 examples):
    - Non-LLM stats: length distribution, boxed rate, think tag rate, coding
      content detection, formatting signals (bold/headers, step markers, etc.)
    - Gemini Flash: analyze all 50 examples in one prompt, extract dataset-level
      rubric properties (style, verbosity, structure, difficulty, content type)

  Phase 2 (500 examples):
    - Non-LLM stats on the larger sample (same metrics)
    - Gemini Flash Lite: batched consistency check — do these samples match the
      rubric extracted in Phase 1?

Output: logs/dataset_research/<dataset>.json

Usage:
    python scripts/data/analyze_dataset.py --dataset mixture_of_thoughts
    python scripts/data/analyze_dataset.py --dataset acemath --phase1-only
    python scripts/data/analyze_dataset.py --dataset acemath --no-llm
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _gemini_client():
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to .env")
    return genai.Client(api_key=api_key)


def gemini_generate(prompt: str, model_env_var: str = "GEMINI_FLASH_MODEL", retries: int = 3) -> str:
    """Call Gemini and return the response text. Retries on transient errors."""
    model = os.environ.get(model_env_var, "gemini-2.0-flash")
    client = _gemini_client()
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  [gemini] error (attempt {attempt+1}): {e} — retrying in {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Non-LLM stats (regex / length based)
# ---------------------------------------------------------------------------

_CODING_PATTERNS = re.compile(
    r"\bdef \w+\(|import \w+|class \w+:|print\(|\.py\b|python\b|javascript\b|leetcode\b|"
    r"Generate an executable|stdin|stdout|#include|int main\(",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\{")
_THINK_TAG = re.compile(r"<think>")
_THINK_CONTENT = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_BOLD_HEADERS = re.compile(r"\*\*.+?\*\*|^#{1,3} ", re.MULTILINE)
_STEP_MARKERS = re.compile(r"\bStep \d+|\b\d+\.\s+\*\*|\*\*Step")
_R1_OPENER = re.compile(r"^(Okay[,.]|Let me |Alright[,.]|Hmm[,.]|So[,.])", re.MULTILINE)


def non_llm_stats(samples: list[dict]) -> dict:
    """Compute regex/length-based stats over a list of {problem, solution} dicts."""
    lengths, sol_lengths = [], []
    boxed, think_tag, coding, bold_headers, step_markers, r1_opener = 0, 0, 0, 0, 0, 0
    multiline_reasoning = 0

    for s in samples:
        prob, sol = s["problem"], s["solution"]
        total = len(prob) + len(sol)
        lengths.append(total)
        sol_lengths.append(len(sol))

        if _BOXED.search(sol):
            boxed += 1
        if _THINK_TAG.search(sol):
            think_tag += 1
        if _CODING_PATTERNS.search(prob) or _CODING_PATTERNS.search(sol):
            coding += 1
        if _BOLD_HEADERS.search(sol):
            bold_headers += 1
        if _STEP_MARKERS.search(sol):
            step_markers += 1
        if _R1_OPENER.search(sol):
            r1_opener += 1
        if sol.count("\n") >= 5:
            multiline_reasoning += 1

    n = len(samples)
    ls = sorted(lengths)

    def pct(p):
        return ls[min(int(p / 100 * n), n - 1)]

    buckets = {
        "0_3k": sum(1 for l in ls if l < 3000),
        "3k_7k": sum(1 for l in ls if 3000 <= l < 7000),
        "7k_14k": sum(1 for l in ls if 7000 <= l < 14000),
        "14k_28k": sum(1 for l in ls if 14000 <= l < 28000),
        "28k_plus": sum(1 for l in ls if l >= 28000),
    }

    # Estimate token counts and survival rates at training seq_len thresholds.
    # Formula matches train.py: chars / 3.5 + 60 tok chat-template overhead.
    _CHARS_PER_TOK = 3.5
    _TEMPLATE_TOK = 60
    tok_estimates = [int(l / _CHARS_PER_TOK) + _TEMPLATE_TOK for l in lengths]
    seq_filter = {}
    for seq_len in (2048, 4096, 8192, 16384):
        kept = sum(1 for t in tok_estimates if t <= seq_len)
        seq_filter[f"seq{seq_len}"] = {
            "kept": kept,
            "dropped": n - kept,
            "survival_rate": round(kept / n, 3),
        }

    return {
        "n": n,
        "length": {
            "mean": round(statistics.mean(lengths)),
            "median": round(statistics.median(lengths)),
            "stdev": round(statistics.stdev(lengths)) if n > 1 else 0,
            "p10": pct(10), "p25": pct(25), "p50": pct(50),
            "p75": pct(75), "p90": pct(90), "p95": pct(95), "p99": pct(99),
        },
        "length_buckets": {k: {"count": v, "pct": round(v / n, 3)} for k, v in buckets.items()},
        "seq_filter": seq_filter,
        "boxed_rate": round(boxed / n, 3),
        "think_tag_rate": round(think_tag / n, 3),
        "coding_content_rate": round(coding / n, 3),
        "bold_headers_rate": round(bold_headers / n, 3),
        "step_markers_rate": round(step_markers / n, 3),
        "r1_opener_rate": round(r1_opener / n, 3),
        "multiline_reasoning_rate": round(multiline_reasoning / n, 3),
    }


# ---------------------------------------------------------------------------
# LLM analysis — Phase 1: extract rubric from 50 examples
# ---------------------------------------------------------------------------

_PHASE1_PROMPT_TEMPLATE = """You are analyzing a math training dataset to extract its properties.
Below are {n} samples from the dataset "{dataset}". Each sample shows the PROBLEM and the START of the SOLUTION (first 400 chars) and END (last 200 chars).

{samples_text}

---
Analyze these samples and return a JSON object with EXACTLY these fields:

{{
  "style_score": <1-10, where 1=fully structured/concise like a textbook, 10=fully R1 exploratory ("Okay let me think...")>,
  "style_label": <"structured" | "semi-structured" | "r1_exploratory">,

  "verbosity_score": <1-10, where 1=minimal (just answer), 10=extremely verbose with lots of prose>,
  "verbosity_label": <"minimal" | "concise" | "moderate" | "verbose" | "very_verbose">,

  "step_by_step_score": <1-10, where 1=no explicit steps, 10=every step is numbered/labeled, like "Step 1: ..., Step 2: ...">,
  "proof_style_score": <1-10, where 1=informal/conversational, 10=fully formal proof with lemmas/QED/rigorous logic>,
  "exploratory_score": <1-10, where 1=linear solution with no backtracking, 10=heavy backtracking and self-correction ("wait, that's wrong... let me reconsider")>,
  "latex_density_score": <1-10, where 1=plain text answers, 10=every expression in LaTeX>,

  "difficulty_score": <1-10, where 1=elementary arithmetic, 4=middle school, 6=high school, 8=AMC/AIME level, 10=IMO/research>,
  "difficulty_label": <"elementary" | "middle_school" | "high_school" | "competition" | "olympiad" | "mixed">,

  "reasoning_type": <"direct_answer" | "step_by_step" | "exploratory_cot" | "proof_style" | "mixed">,
  "structure_features": <list of observed features from: ["bold_headers", "numbered_steps", "latex_heavy", "plain_prose", "backtracking", "self_correction", "scratch_work", "formal_proof", "word_problem_setup"]>,

  "content_types": <list from: ["algebra", "arithmetic", "geometry", "number_theory", "calculus", "combinatorics", "probability", "word_problems", "competition_math", "proof_based"]>,
  "answer_format": <"plain_number" | "boxed_only" | "think_then_boxed" | "multiple_formats" | "no_consistent_format">,
  "think_tag_usage": <"none" | "always" | "sometimes" | "stripped">,

  "is_math_dataset": <true | false>,
  "coding_contamination": <"none" | "some" | "heavy">,
  "quality_notes": <1-2 sentence summary of dataset quality and any concerns>,
  "overall_quality_score": <1-10>
}}

Return ONLY the JSON object, no other text.
"""


def format_samples_for_prompt(samples: list[dict], dataset: str) -> str:
    lines = []
    for i, s in enumerate(samples):
        sol = s["solution"]
        sol_preview = sol[:400] + ("..." if len(sol) > 400 else "")
        sol_end = ("..." + sol[-200:]) if len(sol) > 600 else ""
        lines.append(
            f"[{i+1}] PROBLEM: {s['problem'][:300]}\n"
            f"    SOLUTION START: {sol_preview}\n"
            + (f"    SOLUTION END: {sol_end}\n" if sol_end else "")
        )
    return "\n".join(lines)


def phase1_llm_analysis(samples: list[dict], dataset: str) -> dict:
    print(f"  [phase1-llm] Calling Gemini Flash on {len(samples)} samples...", flush=True)
    samples_text = format_samples_for_prompt(samples, dataset)
    prompt = _PHASE1_PROMPT_TEMPLATE.format(
        n=len(samples), dataset=dataset, samples_text=samples_text
    )
    raw = gemini_generate(prompt, model_env_var="GEMINI_FLASH_MODEL")
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [phase1-llm] JSON parse error: {e}\n  Raw: {raw[:500]}", flush=True)
        result = {"parse_error": str(e), "raw_response": raw[:1000]}
    result["model"] = os.environ.get("GEMINI_FLASH_MODEL", "gemini-2.0-flash")
    return result


# ---------------------------------------------------------------------------
# LLM analysis — Phase 2: consistency check on 500 examples
# ---------------------------------------------------------------------------

_PHASE2_BATCH_PROMPT = """You are checking whether samples from a math dataset match an established rubric.

RUBRIC (from Phase 1 analysis):
{rubric_summary}

Below are {n} samples. For each, respond with a JSON array of objects:
{{
  "idx": <sample index 1-{n}>,
  "is_math": <true|false>,
  "matches_style": <true|false, does it match the rubric style/format?>,
  "quality": <1-5, where 5=excellent math with clear reasoning, 1=garbage/off-topic>,
  "flag": <null | "coding" | "off_topic" | "malformed" | "wrong_format">
}}

SAMPLES:
{samples_text}

Return ONLY the JSON array, no other text.
"""


def phase2_llm_consistency(
    samples: list[dict], rubric: dict, dataset: str, batch_size: int = 25
) -> dict:
    rubric_summary = json.dumps(
        {k: rubric.get(k) for k in [
            "style_label", "verbosity_label", "reasoning_type",
            "step_by_step_score", "proof_style_score", "exploratory_score",
            "difficulty_label", "difficulty_score",
            "answer_format", "think_tag_usage",
        ]},
        indent=2,
    )

    all_results = []
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch = samples[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        print(
            f"  [phase2-llm] Batch {batch_idx+1}/{n_batches} "
            f"({len(batch)} samples)...",
            flush=True,
        )
        samples_text = format_samples_for_prompt(batch, dataset)
        prompt = _PHASE2_BATCH_PROMPT.format(
            rubric_summary=rubric_summary,
            n=len(batch),
            samples_text=samples_text,
        )
        raw = gemini_generate(prompt, model_env_var="GEMINI_FLASH_LITE_MODEL")
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            batch_results = json.loads(raw)
            all_results.extend(batch_results)
        except json.JSONDecodeError as e:
            print(f"  [phase2-llm] Batch parse error: {e}", flush=True)
        time.sleep(0.5)  # gentle rate limiting

    if not all_results:
        return {"error": "no results parsed"}

    n = len(all_results)
    is_math_count = sum(1 for r in all_results if r.get("is_math"))
    matches_style_count = sum(1 for r in all_results if r.get("matches_style"))
    quality_scores = [r.get("quality", 0) for r in all_results if r.get("quality")]
    flags = [r.get("flag") for r in all_results if r.get("flag")]
    flag_counts = {}
    for f in flags:
        flag_counts[f] = flag_counts.get(f, 0) + 1

    quality_dist = {str(i): sum(1 for q in quality_scores if q == i) for i in range(1, 6)}

    return {
        "model": os.environ.get("GEMINI_FLASH_LITE_MODEL", "gemini-2.0-flash-lite"),
        "n_checked": n,
        "is_math_rate": round(is_math_count / n, 3),
        "matches_style_rate": round(matches_style_count / n, 3),
        "avg_quality": round(statistics.mean(quality_scores), 2) if quality_scores else None,
        "quality_distribution": quality_dist,
        "flag_counts": flag_counts,
        "flagged_samples": [r for r in all_results if r.get("flag")],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deduplication analysis
# ---------------------------------------------------------------------------

def dedup_stats(samples: list[dict]) -> dict:
    """Count unique problems and solutions-per-problem distribution.

    Uses first 200 chars of problem text as dedup key (handles minor whitespace).
    """
    problem_counts: dict[str, int] = {}
    for s in samples:
        key = s["problem"].strip()[:200]
        problem_counts[key] = problem_counts.get(key, 0) + 1

    n_total = len(samples)
    counts = sorted(problem_counts.values(), reverse=True)
    n_unique = len(counts)

    if not counts:
        return {}

    dist = {
        "1": sum(1 for c in counts if c == 1),
        "2_5": sum(1 for c in counts if 2 <= c <= 5),
        "6_20": sum(1 for c in counts if 6 <= c <= 20),
        "21_50": sum(1 for c in counts if 21 <= c <= 50),
        "51_plus": sum(1 for c in counts if c > 50),
    }

    top5 = sorted(problem_counts.items(), key=lambda x: -x[1])[:5]

    return {
        "n_total": n_total,
        "n_unique_problems": n_unique,
        "duplication_ratio": round(n_total / n_unique, 2),
        "solutions_per_problem": {
            "min": min(counts),
            "max": max(counts),
            "mean": round(sum(counts) / n_unique, 1),
            "median": counts[n_unique // 2],
        },
        "solutions_distribution": dist,
        "top_problems": [
            {"count": cnt, "problem_prefix": prob[:100]}
            for prob, cnt in top5
        ],
    }


def load_samples(dataset: str, n: int, min_chars: int = -1, max_chars: int = -1) -> list[dict]:
    from scripts.data.normalize_dataset import DATASETS, NORMALIZERS
    from itertools import chain as ichain

    config = DATASETS[dataset]
    normalize = NORMALIZERS[dataset]

    from datasets import load_dataset as hf_load

    if config.get("all_configs"):
        # Multi-config dataset (e.g. MATH with 7 subject areas) — interleave streams
        streams = []
        for cfg_name in config["all_configs"]:
            streams.append(hf_load(
                config["hf_id"], name=cfg_name, split=config["split"],
                trust_remote_code=True, streaming=True,
            ))
        ds = ichain(*streams)
    else:
        ds = hf_load(
            config["hf_id"],
            name=config.get("subset"),
            split=config["split"],
            trust_remote_code=True,
            streaming=True,
            verification_mode="no_checks",
        )

    samples = []
    seen = 0
    for row in ds:
        r = normalize(row)
        if not r or not r["problem"].strip() or not r["solution"].strip():
            continue
        L = len(r["problem"]) + len(r["solution"])
        if min_chars > 0 and L < min_chars:
            continue
        if max_chars > 0 and L > max_chars:
            continue
        samples.append(r)
        seen += 1
        if seen >= n:
            break

    return samples


def main():
    parser = argparse.ArgumentParser(description="Analyze a math dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name from normalize_dataset.py")
    parser.add_argument("--phase1-size", type=int, default=50, help="Samples for Phase 1")
    parser.add_argument("--phase2-size", type=int, default=500, help="Samples for Phase 2")
    parser.add_argument("--phase1-only", action="store_true", help="Skip Phase 2")
    parser.add_argument("--no-llm", action="store_true", help="Skip all LLM calls (stats only)")
    parser.add_argument(
        "--dedup-size", type=int, default=0,
        help="Load this many samples for dedup analysis (unique problems / solutions per problem). "
             "Use a large value for repetitive datasets (e.g. --dedup-size 50000 for dartmath). "
             "0 = reuse phase2 samples only.",
    )
    parser.add_argument("--min-chars", type=int, default=-1, help="Only include samples with total chars >= this")
    parser.add_argument("--max-chars", type=int, default=-1, help="Only include samples with total chars <= this")
    parser.add_argument(
        "--output-dir", default=str(PROJECT_ROOT / "logs" / "dataset_research"),
        help="Directory to write JSON results",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Include length bucket in filename when filtering
    bucket_suffix = ""
    if args.min_chars > 0 or args.max_chars > 0:
        lo = f"{args.min_chars // 1000}k" if args.min_chars > 0 else "0"
        hi = f"{args.max_chars // 1000}k" if args.max_chars > 0 else "inf"
        bucket_suffix = f"_{lo}-{hi}"
    out_path = out_dir / f"{args.dataset}{bucket_suffix}.json"

    label = args.dataset + (f" [{bucket_suffix.strip('_')}]" if bucket_suffix else "")
    print(f"=== Analyzing: {label} ===", flush=True)
    if args.min_chars > 0 or args.max_chars > 0:
        print(f"  Length filter: min={args.min_chars if args.min_chars > 0 else 'none'}  max={args.max_chars if args.max_chars > 0 else 'none'}", flush=True)

    # --- Phase 1 samples ---
    print(f"\n[Phase 1] Loading {args.phase1_size} samples...", flush=True)
    p1_samples = load_samples(args.dataset, args.phase1_size, args.min_chars, args.max_chars)
    print(f"  Loaded {len(p1_samples)} samples", flush=True)

    print("[Phase 1] Computing non-LLM stats...", flush=True)
    p1_stats = non_llm_stats(p1_samples)
    sf = p1_stats["seq_filter"]
    print(
        f"  boxed={p1_stats['boxed_rate']:.0%}  think={p1_stats['think_tag_rate']:.0%}  "
        f"coding={p1_stats['coding_content_rate']:.0%}  "
        f"r1_opener={p1_stats['r1_opener_rate']:.0%}  "
        f"bold_headers={p1_stats['bold_headers_rate']:.0%}",
        flush=True,
    )
    print(
        f"  seq_filter: seq2048={sf['seq2048']['survival_rate']:.0%}  "
        f"seq4096={sf['seq4096']['survival_rate']:.0%}  "
        f"seq8192={sf['seq8192']['survival_rate']:.0%}",
        flush=True,
    )

    p1_llm = None
    if not args.no_llm:
        p1_llm = phase1_llm_analysis(p1_samples, args.dataset)
        print(
            f"  style={p1_llm.get('style_label')}  "
            f"verbosity={p1_llm.get('verbosity_label')}  "
            f"reasoning={p1_llm.get('reasoning_type')}  "
            f"quality={p1_llm.get('overall_quality_score')}",
            flush=True,
        )

    # --- Phase 2: 500 samples ---
    p2_samples: list[dict] = []
    p2_stats = None
    p2_consistency = None

    if not args.phase1_only:
        print(f"\n[Phase 2] Loading {args.phase2_size} samples...", flush=True)
        p2_samples = load_samples(args.dataset, args.phase2_size, args.min_chars, args.max_chars)
        print(f"  Loaded {len(p2_samples)} samples", flush=True)

        print("[Phase 2] Computing non-LLM stats...", flush=True)
        p2_stats = non_llm_stats(p2_samples)
        print(
            f"  boxed={p2_stats['boxed_rate']:.0%}  think={p2_stats['think_tag_rate']:.0%}  "
            f"coding={p2_stats['coding_content_rate']:.0%}  "
            f"r1_opener={p2_stats['r1_opener_rate']:.0%}  "
            f"bold_headers={p2_stats['bold_headers_rate']:.0%}",
            flush=True,
        )

        if not args.no_llm and p1_llm and "parse_error" not in p1_llm:
            p2_consistency = phase2_llm_consistency(p2_samples, p1_llm, args.dataset)
            print(
                f"  is_math={p2_consistency.get('is_math_rate'):.0%}  "
                f"matches_style={p2_consistency.get('matches_style_rate'):.0%}  "
                f"avg_quality={p2_consistency.get('avg_quality')}",
                flush=True,
            )

    # --- Dedup analysis ---
    dedup = None
    if not args.phase1_only:
        if args.dedup_size > 0 and args.dedup_size > args.phase2_size:
            print(f"\n[Dedup] Loading {args.dedup_size} samples for dedup analysis...", flush=True)
            dedup_samples = load_samples(args.dataset, args.dedup_size, args.min_chars, args.max_chars)
            print(f"  Loaded {len(dedup_samples)} samples", flush=True)
        else:
            dedup_samples = p2_samples

        dedup = dedup_stats(dedup_samples)
        print(
            f"[Dedup] unique={dedup['n_unique_problems']}  "
            f"total={dedup['n_total']}  "
            f"ratio={dedup['duplication_ratio']}x  "
            f"sols/prob: min={dedup['solutions_per_problem']['min']} "
            f"mean={dedup['solutions_per_problem']['mean']} "
            f"max={dedup['solutions_per_problem']['max']}",
            flush=True,
        )

    # --- Build output ---
    result = {
        "dataset": args.dataset,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "length_filter": {
            "min_chars": args.min_chars if args.min_chars > 0 else None,
            "max_chars": args.max_chars if args.max_chars > 0 else None,
        },
        "phase1": {
            "sample_size": len(p1_samples),
            "non_llm_stats": p1_stats,
            "llm_analysis": p1_llm,
        },
        "phase2": {
            "sample_size": len(p2_samples) if p2_stats else 0,
            "non_llm_stats": p2_stats,
            "consistency_check": p2_consistency,
        } if not args.phase1_only else None,
        "dedup": dedup,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
