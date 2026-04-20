"""
VHELM-Aligned Fairness & Robustness Analysis
=============================================
Addresses Reviewer 1 (zuah) concern about VHELM alignment.

Experiment 1: Performance Disparity (Fairness) — ZERO API calls
Experiment 2: Visual Perturbation (Robustness) — 50 images x 1 model

Output: Table X for rebuttal + violation breakdown
"""

import os
import json
import base64
import time
import random
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from google import genai
from google.genai import types

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = os.environ.get("PLANTINQUIRY_HOME", ".")
LLM_JUDGE_DIR = os.path.join(BASE_DIR, "eval/llm_judge_results")
DATASET_CSV = os.path.join(BASE_DIR, "dataset/plantinquiryvqa_test_subset.csv")
IMAGE_DIR = os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval/vhelm_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Majority crops: Top 5 most frequent crops in judge results (32-23 judgments each)
MAJORITY_CROPS = ["Mango", "Pepper", "Papaya", "Cotton", "Jackfruit"]

# Minority crops: Bottom 5 least frequent crops in judge results (2-6 judgments each)
MINORITY_CROPS = ["Lemon", "Bitter Gourd", "Cucumber", "Eggplant Brinjal", "Rubber"]

# Models to analyze (must have llm_judge_results)
MODELS_TO_ANALYZE = [
    "Gemini-3-Flash",
    "Qwen3-VL-235B",
    "Qwen3-VL-32B",
    "Seed-1.6-Flash",
    "Gemini-2.5-Pro",
    "Gemini-2.5-Flash",
]

# ==============================================================================
# EXPERIMENT 1: FAIRNESS (Performance Disparity) — Zero API calls
# ==============================================================================

def load_judge_results(model_name):
    """Load LLM judge results for a model."""
    path = os.path.join(LLM_JUDGE_DIR, model_name, "judge_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("results", [])


def compute_group_scores(results, crop_list):
    """Compute average scores for a group of crops."""
    crop_set = set(c.lower() for c in crop_list)

    scores = {"diagnostic_correctness": [], "clinical_completeness": [],
              "reasoning_quality": [], "safety_actionability": [],
              "visual_grounding": []}

    for r in results:
        crop = str(r.get("crop", "")).lower()
        if crop in crop_set:
            s = r.get("scores", {})
            for dim in scores:
                val = s.get(dim)
                if val is not None:
                    scores[dim].append(val)

    # Compute averages
    avgs = {}
    n = 0
    for dim in scores:
        if scores[dim]:
            avgs[dim] = sum(scores[dim]) / len(scores[dim])
            n = max(n, len(scores[dim]))
        else:
            avgs[dim] = 0.0

    avgs["n_judgments"] = n
    score_vals = [v for k, v in avgs.items() if k != "n_judgments" and isinstance(v, (int, float))]
    avgs["overall"] = sum(score_vals) / len(score_vals) if score_vals else 0.0
    return avgs


def run_fairness_analysis():
    """Experiment 1: Performance Disparity across Majority vs Minority crop groups."""
    print("=" * 70)
    print("EXPERIMENT 1: FAIRNESS — Performance Disparity (VHELM Aligned)")
    print("=" * 70)
    print(f"\nMajority crops (high-freq): {MAJORITY_CROPS}")
    print(f"Minority crops (low-freq):  {MINORITY_CROPS}")

    results_table = []

    for model_name in MODELS_TO_ANALYZE:
        results = load_judge_results(model_name)
        if results is None:
            print(f"  SKIP {model_name}: no judge results found")
            continue

        maj_scores = compute_group_scores(results, MAJORITY_CROPS)
        min_scores = compute_group_scores(results, MINORITY_CROPS)

        if maj_scores["n_judgments"] == 0 or min_scores["n_judgments"] == 0:
            print(f"  SKIP {model_name}: insufficient data (maj={maj_scores['n_judgments']}, min={min_scores['n_judgments']})")
            continue

        gap = maj_scores["overall"] - min_scores["overall"]
        maj_diag = maj_scores.get("diagnostic_correctness", 0) or 0
        min_diag = min_scores.get("diagnostic_correctness", 0) or 0
        maj_safe = maj_scores.get("safety_actionability", 0) or 0
        min_safe = min_scores.get("safety_actionability", 0) or 0

        results_table.append({
            "model": model_name,
            "majority_score": maj_scores["overall"],
            "minority_score": min_scores["overall"],
            "fairness_gap": gap,
            "majority_n": maj_scores["n_judgments"],
            "minority_n": min_scores["n_judgments"],
            "majority_diag": maj_diag,
            "minority_diag": min_diag,
            "majority_safe": maj_safe,
            "minority_safe": min_safe,
        })

    # Print Table
    print(f"\n{'Model':<20} {'Maj (Overall)':>14} {'Min (Overall)':>14} {'Δ Gap':>8} {'Maj S_diag':>10} {'Min S_diag':>10} {'Maj n':>6} {'Min n':>6}")
    print("-" * 90)
    for r in results_table:
        print(f"{r['model']:<20} {r['majority_score']:>14.3f} {r['minority_score']:>14.3f} {r['fairness_gap']:>+8.3f} {r['majority_diag']:>10.3f} {r['minority_diag']:>10.3f} {r['majority_n']:>6} {r['minority_n']:>6}")

    # Average fairness gap
    avg_gap = np.mean([r["fairness_gap"] for r in results_table]) if results_table else 0
    print(f"\nAverage Fairness Gap (Δ): {avg_gap:+.3f}")
    print(f"Interpretation: {'Small gap — models generalize well across crop types' if abs(avg_gap) < 0.5 else 'Notable disparity — models favor majority crops'}")

    return results_table


# ==============================================================================
# EXPERIMENT 2: ROBUSTNESS (Visual Perturbation) — Gemini-3-Flash only
# ==============================================================================

def apply_perturbations(image_path, output_dir):
    """Apply 3 visual perturbations to simulate field conditions."""
    img = cv2.imread(image_path)
    if img is None:
        return {}

    results = {}

    # 1. Gaussian Blur (sigma=3) - Simulates out-of-focus field camera
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    blur_path = os.path.join(output_dir, "blur_" + os.path.basename(image_path))
    cv2.imwrite(blur_path, blurred)
    results["blur"] = blur_path

    # 2. Random Crop (85% center crop) - Simulates incomplete framing
    h, w = img.shape[:2]
    margin_h, margin_w = int(h * 0.075), int(w * 0.075)
    cropped = img[margin_h:h-margin_h, margin_w:w-margin_w]
    cropped = cv2.resize(cropped, (w, h))
    crop_path = os.path.join(output_dir, "crop_" + os.path.basename(image_path))
    cv2.imwrite(crop_path, cropped)
    results["crop"] = crop_path

    # 3. Brightness Shift (simulate overexposure in field sunlight)
    bright = cv2.convertScaleAbs(img, alpha=1.4, beta=40)
    bright_path = os.path.join(output_dir, "bright_" + os.path.basename(image_path))
    cv2.imwrite(bright_path, bright)
    results["bright"] = bright_path

    return results


def retry_with_backoff(func, *args, retries=3, backoff=5, **kwargs):
    """Simple retry wrapper."""
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                return f"ERROR: {e}"
            time.sleep(backoff * (2 ** attempt) + random.uniform(0, 1))


def call_gemini_for_diagnosis(image_path, prompt):
    """Call Gemini to diagnose from image."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            types.Part.from_bytes(data=image_data, mime_type='image/jpeg'),
            prompt
        ]
    )
    return response.text.strip()


def judge_diagnostic_match(response_text, gt_disease, gt_crop):
    """Quick Gemini judge: did the model correctly identify crop and disease?"""
    prompt = f"""You are evaluating a plant diagnosis model. 

Ground Truth:
- Crop: {gt_crop}
- Disease: {gt_disease}

Model Response:
\"\"\"{response_text[:1500]}\"\"\"

Did the model correctly identify:
1. The correct crop (or a reasonable synonym)?
2. The correct disease (or a close match)?

Output ONLY:
CROP_CORRECT: [YES/NO]
DISEASE_CORRECT: [YES/NO]
OVERALL_SCORE: [1-5]
"""
    resp = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    text = resp.text

    crop_match = re.search(r"CROP_CORRECT:\s*(YES|NO)", text, re.IGNORECASE)
    disease_match = re.search(r"DISEASE_CORRECT:\s*(YES|NO)", text, re.IGNORECASE)
    score_match = re.search(r"OVERALL_SCORE:\s*(\d)", text)

    return {
        "crop_correct": crop_match and crop_match.group(1).upper() == "YES",
        "disease_correct": disease_match and disease_match.group(1).upper() == "YES",
        "score": int(score_match.group(1)) if score_match else 1
    }


DIAGNOSIS_PROMPT = """You are a plant pathologist. Look at this image of a plant leaf and provide your complete diagnostic assessment.

Include:
1. Crop identification
2. Whether it is healthy or diseased
3. If diseased: the specific disease name, severity level, and key visual symptoms
4. Recommended treatment

Provide your full diagnosis."""


def run_robustness_analysis():
    """Experiment 2: Visual Perturbation Robustness."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ROBUSTNESS — Visual Perturbation (VHELM Aligned)")
    print("=" * 70)
    print("Model: Gemini-3-Flash | Perturbations: Blur, Crop, Brightness")

    # Load CSV to get minority group images
    import pandas as pd
    df = pd.read_csv(DATASET_CSV)

    # Get minority group images (by crop)
    minority_set = set(c.lower() for c in MINORITY_CROPS)
    minority_imgs = df[df['crop'].str.lower().isin(minority_set)]
    minority_unique = minority_imgs[['image_id', 'crop', 'disease']].drop_duplicates(subset='image_id')

    # Filter for existing images
    valid = []
    for _, row in minority_unique.iterrows():
        img_path = os.path.join(IMAGE_DIR, row['image_id'])
        if os.path.exists(img_path):
            valid.append(row)
    minority_unique = pd.DataFrame(valid)

    n_samples = min(50, len(minority_unique))
    if n_samples < 5:
        # If minority has too few images, supplement with some majority
        print(f"  Only {len(minority_unique)} minority images. Supplementing with other images...")
        all_imgs = df[['image_id', 'crop', 'disease']].drop_duplicates(subset='image_id')
        valid_all = []
        for _, row in all_imgs.iterrows():
            if os.path.exists(os.path.join(IMAGE_DIR, row['image_id'])):
                valid_all.append(row)
        all_valid = pd.DataFrame(valid_all)
        minority_unique = all_valid.head(50)
        n_samples = min(50, len(minority_unique))

    print(f"  Using {n_samples} images for robustness testing")

    # Create perturbation directory
    perturb_dir = os.path.join(OUTPUT_DIR, "perturbed_images")
    os.makedirs(perturb_dir, exist_ok=True)

    # Run evaluation
    results = {"clean": [], "blur": [], "crop": [], "bright": []}
    detailed = []

    for i, (_, row) in enumerate(minority_unique.head(n_samples).iterrows()):
        image_id = row['image_id']
        gt_crop = row['crop']
        gt_disease = row['disease']
        img_path = os.path.join(IMAGE_DIR, image_id)

        print(f"  [{i+1}/{n_samples}] {image_id[:12]}... ({gt_crop}/{gt_disease})")

        # Apply perturbations
        perturbed_paths = apply_perturbations(img_path, perturb_dir)

        # Test each condition
        conditions = {"clean": img_path}
        conditions.update(perturbed_paths)

        entry = {"image_id": image_id, "crop": gt_crop, "disease": gt_disease}

        for condition, cond_path in conditions.items():
            # Get diagnosis
            response = retry_with_backoff(
                call_gemini_for_diagnosis, cond_path, DIAGNOSIS_PROMPT
            )

            time.sleep(0.5)

            # Judge
            if str(response).startswith("ERROR"):
                judgment = {"crop_correct": False, "disease_correct": False, "score": 1}
            else:
                judgment = retry_with_backoff(
                    judge_diagnostic_match, response, gt_disease, gt_crop
                )
                if isinstance(judgment, str):  # Error
                    judgment = {"crop_correct": False, "disease_correct": False, "score": 1}

            time.sleep(0.5)

            results[condition].append(judgment)
            entry[f"{condition}_score"] = judgment["score"] if isinstance(judgment, dict) else 1
            entry[f"{condition}_crop"] = judgment.get("crop_correct", False)
            entry[f"{condition}_disease"] = judgment.get("disease_correct", False)

        detailed.append(entry)

    # Compute summary
    print(f"\n{'Condition':<12} {'Avg Score':>10} {'Crop Acc':>10} {'Disease Acc':>12} {'n':>5}")
    print("-" * 55)

    robustness_summary = {}
    for condition in ["clean", "blur", "crop", "bright"]:
        if not results[condition]:
            continue
        scores = [r["score"] for r in results[condition] if isinstance(r, dict)]
        crop_acc = sum(1 for r in results[condition] if isinstance(r, dict) and r.get("crop_correct")) / len(results[condition]) * 100
        dis_acc = sum(1 for r in results[condition] if isinstance(r, dict) and r.get("disease_correct")) / len(results[condition]) * 100
        avg_score = sum(scores) / len(scores) if scores else 0

        print(f"{condition:<12} {avg_score:>10.2f} {crop_acc:>9.1f}% {dis_acc:>11.1f}% {len(scores):>5}")
        robustness_summary[condition] = {
            "avg_score": avg_score,
            "crop_accuracy": crop_acc,
            "disease_accuracy": dis_acc,
            "n": len(scores)
        }

    # Compute drops
    if "clean" in robustness_summary:
        clean = robustness_summary["clean"]
        print(f"\n--- Robustness Drop (vs Clean baseline) ---")
        for cond in ["blur", "crop", "bright"]:
            if cond in robustness_summary:
                drop = clean["avg_score"] - robustness_summary[cond]["avg_score"]
                drop_pct = (drop / clean["avg_score"]) * 100 if clean["avg_score"] > 0 else 0
                dis_drop = clean["disease_accuracy"] - robustness_summary[cond]["disease_accuracy"]
                print(f"  {cond:<10}: Score drop = {drop:+.2f} ({drop_pct:+.1f}%), Disease Acc drop = {dis_drop:+.1f}%")

    return robustness_summary, detailed


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║     VHELM-ALIGNED FAIRNESS & ROBUSTNESS ANALYSIS                   ║")
    print("║     Addressing Reviewer 1 (zuah) — Performance Disparity & Bias    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    # --- Experiment 1: Fairness (no API calls) ---
    fairness_results = run_fairness_analysis()

    # --- Experiment 2: Robustness (50 images x 1 model) ---
    robustness_results, robustness_detailed = run_robustness_analysis()

    # --- Save all results ---
    output = {
        "experiment_1_fairness": fairness_results,
        "experiment_2_robustness": {
            "summary": robustness_results,
            "detailed": robustness_detailed
        },
        "majority_crops": MAJORITY_CROPS,
        "minority_crops": MINORITY_CROPS,
    }
    output_path = os.path.join(OUTPUT_DIR, "vhelm_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n\nAll results saved to: {output_path}")

    # --- Print Rebuttal Table ---
    print("\n" + "=" * 85)
    print("TABLE X: Fairness & Robustness Spot-Check (VHELM Aligned)")
    print("=" * 85)
    print(f"{'Model':<20} {'Maj S_overall':>13} {'Min S_overall':>13} {'Δ Fairness':>11} {'Robust (Blur)':>14} {'Robust (Crop)':>14}")
    print("-" * 85)
    for r in fairness_results:
        model = r["model"]
        # Get robustness data (only for Gemini-3-Flash)
        if model == "Gemini-3-Flash" and robustness_results:
            blur_score = robustness_results.get("blur", {}).get("avg_score", "—")
            crop_score = robustness_results.get("crop", {}).get("avg_score", "—")
            blur_str = f"{blur_score:.2f}" if isinstance(blur_score, float) else blur_score
            crop_str = f"{crop_score:.2f}" if isinstance(crop_score, float) else crop_score
        else:
            blur_str = "—"
            crop_str = "—"

        print(f"{model:<20} {r['majority_score']:>13.3f} {r['minority_score']:>13.3f} {r['fairness_gap']:>+11.3f} {blur_str:>14} {crop_str:>14}")

    print(f"\nNote: Robustness tested on Gemini-3-Flash only (top model). Δ Fairness = Majority − Minority.")
    print(f"      Positive Δ → model favors majority crops. Closer to 0 → more equitable.")


if __name__ == "__main__":
    main()
