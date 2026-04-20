"""
Compute Prevalence Bias (B, paper Eq. 7) for all 18 models.

Method:
  1. Build empirical corpus frequency f(d) from the full dataset CSV.
  2. For each model's judge_results.json:
     * Treat diagnostic_correctness < 4 as a 'misdiagnosis'.
     * Extract the *predicted* disease by scanning model_answer for known
       disease keywords; pick the match whose canonical entry has the highest
       corpus frequency (most-likely what the model asserted).
     * A misdiagnosis contributes +1 to the numerator iff f(pred) > f(GT).
  3. B = above / total_misdiagnoses.
     B = 0.5  → no prevalence preference in errors.
     B > 0.5  → model falls back to common diseases (long-tail penalty).
     B < 0.5  → model inversely biased (rare over common).
"""

import csv
import json
import os
import re
from collections import Counter

BASE = os.environ.get("PLANTINQUIRY_HOME", ".")
JUDGE_DIR = os.path.join(BASE, "eval/llm_judge_results")
DATASET_CSV = os.path.join(BASE, "dataset/plantinquiryvqa_test.csv")
TRAIN_CSV   = os.path.join(BASE, "dataset/plantinquiryvqa_train.csv")
OUT_PATH = os.path.join(BASE, "eval/prevalence_bias_all_18_models.json")

MODELS = [
    "Gemini-3-Flash", "Gemini-2.5-Pro", "Qwen3-VL-235B", "Seed-1.6-Flash",
    "Llama-3.2-90B-Vision", "Llama-4-Maverick", "Gemini-2.5-Flash",
    "Qwen3-VL-32B", "Gemma-3-27B", "Pixtral-12B", "Qwen2.5-VL-32B",
    "Phi-4-Multimodal", "Qwen2.5-VL-72B", "Grok-4.1-Fast",
    "Mistral-Medium-3.1", "Ministral-8B", "Ministral-3B", "Qwen-VL-Plus",
]


def build_frequency():
    """Empirical corpus frequency f(d) over disease labels (unique-image level)."""
    freq = Counter()
    seen = set()
    for fn in [DATASET_CSV, TRAIN_CSV]:
        if not os.path.exists(fn):
            continue
        with open(fn) as f:
            for r in csv.DictReader(f):
                if r["image_id"] in seen:
                    continue
                seen.add(r["image_id"])
                d = str(r.get("disease", "")).strip()
                if d and d.lower() not in ("nan", ""):
                    freq[d.lower()] += 1
    return freq


def extract_predicted(answer: str, freq: Counter):
    """Return the disease string from `answer` with the highest corpus freq,
    or None if no known disease name is mentioned."""
    a = answer.lower()
    best_name, best_f = None, -1
    for name, f in freq.items():
        # whole-word match to avoid partials like 'rot' inside 'rotate'
        pattern = r"\b" + re.escape(name) + r"\b"
        if re.search(pattern, a):
            if f > best_f:
                best_f, best_name = f, name
    return best_name, best_f


def compute_B(results, freq):
    misdiag, above = 0, 0
    evaluable = 0
    for r in results:
        s = r.get("scores") or {}
        dc = s.get("diagnostic_correctness")
        if dc is None or dc >= 4:            # not a misdiagnosis
            continue
        gt = str(r.get("disease", "")).strip().lower()
        if not gt or gt == "nan":
            continue
        pred, f_pred = extract_predicted(str(r.get("model_answer", "")), freq)
        if pred is None or pred == gt:
            continue
        misdiag += 1
        f_gt = freq.get(gt, 0)
        if f_pred > f_gt:
            above += 1
        evaluable += 1
    return {
        "n_misdiagnoses": misdiag,
        "n_evaluable":    evaluable,
        "n_more_frequent_errors": above,
        "B": round(above / misdiag, 3) if misdiag else None,
    }


def main():
    freq = build_frequency()
    print(f"Corpus frequency built over {sum(freq.values())} images "
          f"across {len(freq)} disease classes")
    print(f"{'Model':<22} {'n_mis':>7} {'n>f(GT)':>8} {'B':>6}  Interpretation")
    print("-" * 70)
    results_table = {}
    for m in MODELS:
        p = os.path.join(JUDGE_DIR, m, "judge_results.json")
        if not os.path.exists(p):
            print(f"{m:<22}  (judge results missing)")
            continue
        with open(p) as f:
            data = json.load(f).get("results", [])
        r = compute_B(data, freq)
        results_table[m] = r
        if r["B"] is None:
            print(f"{m:<22}  (no misdiagnoses scored)")
            continue
        interp = ("fallback-to-common" if r["B"] > 0.55 else
                  "inverse-bias"       if r["B"] < 0.45 else
                  "balanced")
        print(f"{m:<22} {r['n_misdiagnoses']:>7} {r['n_more_frequent_errors']:>8} "
              f"{r['B']:>6.3f}  {interp}")
    with open(OUT_PATH, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
