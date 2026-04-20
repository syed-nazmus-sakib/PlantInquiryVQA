"""
Extend the VHELM-aligned Cross-Class Fairness table (F) to all 18 evaluated models.

Method (same as eval/vhelm_fairness_robustness.py):
  * Majority crops (high-frequency):  Mango, Pepper, Papaya, Cotton, Jackfruit
  * Minority crops (low-frequency):   Lemon, Bitter Gourd, Cucumber,
                                       Eggplant Brinjal, Rubber
  * For each model we load its judge_results.json (LLM-as-judge, 1-5 scale),
    split samples by crop group, and average the five dimensions:
        diagnostic_correctness, clinical_completeness, reasoning_quality,
        safety_actionability, visual_grounding.
  * The "overall" column reproduces the rebuttal table ('S_clin mean per split');
    ΔGap = Majority - Minority  (paper Eq. 8).

Also reports Prevalence Bias (B) by comparing each misdiagnosis to the empirical
crop-disease frequency distribution inferred from the dataset CSV.
"""

import json
import os
import csv
from collections import Counter

BASE_DIR = os.environ.get("PLANTINQUIRY_HOME", ".")
JUDGE_DIR = os.path.join(BASE_DIR, "eval/llm_judge_results")
DATASET_CSV = os.path.join(BASE_DIR, "dataset/plantinquiryvqa_test_subset.csv")
OUT_PATH = os.path.join(BASE_DIR, "eval/vhelm_analysis/fairness_all_18_models.json")

MAJORITY = {"mango", "pepper", "papaya", "cotton", "jackfruit"}
MINORITY = {"lemon", "bitter gourd", "cucumber", "eggplant brinjal", "rubber"}

MODELS = [
    "Gemini-3-Flash", "Gemini-2.5-Pro", "Qwen3-VL-235B", "Seed-1.6-Flash",
    "Llama-3.2-90B-Vision", "Llama-4-Maverick", "Gemini-2.5-Flash",
    "Qwen3-VL-32B", "Gemma-3-27B", "Pixtral-12B", "Qwen2.5-VL-32B",
    "Phi-4-Multimodal", "Qwen2.5-VL-72B", "Grok-4.1-Fast",
    "Mistral-Medium-3.1", "Ministral-8B", "Ministral-3B", "Qwen-VL-Plus",
]

DIMS = ["diagnostic_correctness", "clinical_completeness", "reasoning_quality",
        "safety_actionability", "visual_grounding"]


def load_judge(model):
    p = os.path.join(JUDGE_DIR, model, "judge_results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get("results", [])


def group_stats(results, crop_set):
    bucket = {d: [] for d in DIMS}
    n = 0
    for r in results:
        crop = str(r.get("crop", "")).lower().strip()
        if crop in crop_set:
            n += 1
            s = r.get("scores") or {}
            for d in DIMS:
                v = s.get(d)
                if v is not None:
                    bucket[d].append(float(v))
    means = {d: (sum(v)/len(v) if v else 0.0) for d, v in bucket.items()}
    means["overall"] = sum(means[d] for d in DIMS) / len(DIMS) if n else 0.0
    means["n"] = n
    return means


def classify(gap):
    if abs(gap) < 0.1:
        return "Fair (ΔF ≈ 0)"
    if gap >= 0.3:
        return "Minority Penalty"
    if gap <= -0.3:
        return "Minority Benefit"
    return "Low Disparity"


def compute_prevalence_bias(results, disease_freq):
    """Fraction of misdiagnoses where predicted disease is more frequent than GT."""
    if not disease_freq:
        return None
    total, above = 0, 0
    for r in results:
        s = r.get("scores") or {}
        dc = s.get("diagnostic_correctness")
        # 'diagnostic_correctness' < 4 → misdiagnosis
        if dc is None or dc >= 4:
            continue
        gt = str(r.get("disease", "")).lower().strip()
        pred = str(r.get("predicted_disease", r.get("prediction", ""))).lower().strip()
        if not gt or not pred:
            continue
        f_gt = disease_freq.get(gt, 0)
        f_pr = disease_freq.get(pred, 0)
        if f_gt == 0 and f_pr == 0:
            continue
        total += 1
        if f_pr > f_gt:
            above += 1
    return (above / total) if total else None


def build_disease_frequency():
    freq = Counter()
    if not os.path.exists(DATASET_CSV):
        return freq
    with open(DATASET_CSV) as f:
        for row in csv.DictReader(f):
            d = str(row.get("disease", "")).lower().strip()
            if d and d != "nan":
                freq[d] += 1
    return freq


def main():
    disease_freq = build_disease_frequency()
    rows = []
    print("=" * 110)
    print(f"{'Model':<22} {'Maj S_clin':>11} {'Min S_clin':>11} {'ΔF Gap':>8} "
          f"{'Maj S_dis':>10} {'Min S_dis':>10} {'Maj Safe':>9} {'Min Safe':>9} "
          f"{'Maj n':>6} {'Min n':>6}  Interp.")
    print("-" * 110)
    maj_means = []
    min_means = []
    gaps = []
    for m in MODELS:
        res = load_judge(m)
        if res is None:
            print(f"{m:<22}  (no judge results found)")
            continue
        maj = group_stats(res, MAJORITY)
        mn  = group_stats(res, MINORITY)
        if maj["n"] == 0 or mn["n"] == 0:
            print(f"{m:<22}  (insufficient maj/min samples)")
            continue
        gap = maj["overall"] - mn["overall"]
        interp = classify(gap)
        prev_b = compute_prevalence_bias(res, disease_freq)
        rows.append({
            "model": m,
            "majority_overall": round(maj["overall"], 3),
            "minority_overall": round(mn["overall"], 3),
            "delta_F_gap":      round(gap, 3),
            "majority_diag":    round(maj["diagnostic_correctness"], 3),
            "minority_diag":    round(mn["diagnostic_correctness"], 3),
            "majority_safe":    round(maj["safety_actionability"], 3),
            "minority_safe":    round(mn["safety_actionability"], 3),
            "majority_n":       maj["n"],
            "minority_n":       mn["n"],
            "prevalence_B":     round(prev_b, 3) if prev_b is not None else None,
            "interpretation":   interp,
        })
        maj_means.append(maj["overall"])
        min_means.append(mn["overall"])
        gaps.append(gap)
        print(f"{m:<22} {maj['overall']:>11.3f} {mn['overall']:>11.3f} {gap:>+8.3f} "
              f"{maj['diagnostic_correctness']:>10.3f} {mn['diagnostic_correctness']:>10.3f} "
              f"{maj['safety_actionability']:>9.3f} {mn['safety_actionability']:>9.3f} "
              f"{maj['n']:>6} {mn['n']:>6}  {interp}")
    print("-" * 110)
    if rows:
        avg_maj = sum(maj_means)/len(maj_means)
        avg_min = sum(min_means)/len(min_means)
        avg_gap = sum(gaps)/len(gaps)
        print(f"{'Average (18 models)':<22} {avg_maj:>11.3f} {avg_min:>11.3f} {avg_gap:>+8.3f}")
    with open(OUT_PATH, "w") as f:
        json.dump({"rows": rows,
                   "majority_crops": sorted(MAJORITY),
                   "minority_crops": sorted(MINORITY)},
                  f, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
