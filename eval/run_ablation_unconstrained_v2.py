import os
import json
import base64
import pandas as pd
import time
import requests
import random
from google import genai
from google.genai import types
from tqdm import tqdm
from collections import defaultdict
import re

# ==============================================================================
# CONFIG & API KEYS
# ==============================================================================

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# Paths
BASE_DIR = os.environ.get("PLANTINQUIRY_HOME", ".")
DATASET_PATH = os.path.join(BASE_DIR, "dataset/plantinquiryvqa_test_subset.csv")
IMAGE_DIRS = [
    os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images"),
    os.path.join(BASE_DIR, "images"),
]
OUTPUT_FILE = os.path.join(BASE_DIR, "eval/ablation_unconstrained_v2_results.json")

# Models
MODELS = {
    "gemini3_flash": {
        "type": "gemini",
        "model_id": "gemini-3-flash-preview"
    },
    "qwen3_235b": {
        "type": "openrouter",
        "model_id": "qwen/qwen3-vl-235b-a22b-instruct"
    },
    "seed_1_6": {
        "type": "openrouter",
        "model_id": "bytedance-seed/seed-1.6-flash"
    }
}

# =============================================================================
# STRICT UNCONSTRAINED PROMPT
# No CoI scaffold, no question sequence, no template guidance.
# The model must diagnose from scratch in a SINGLE turn.
# =============================================================================
UNCONSTRAINED_PROMPT = """You are a plant pathologist. Look at this image of a plant leaf and provide your complete diagnostic assessment in a single response.

Your response must include:
1. Crop identification
2. Whether it is healthy or diseased
3. If diseased: the specific disease name, severity level (mild/moderate/severe), observed visual symptoms, and recommended treatment
4. If healthy: confirmation with supporting visual evidence

Do NOT ask follow-up questions. Provide your full diagnosis now."""

N_SAMPLES = 50

# ==============================================================================
# CLIENT SETUP
# ==============================================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_image(image_id):
    for img_dir in IMAGE_DIRS:
        path = os.path.join(img_dir, image_id)
        if os.path.exists(path):
            return path
    return None

# ==============================================================================
# RETRY DECORATOR
# ==============================================================================

def retry_with_backoff(retries=5, backoff_in_seconds=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        print(f"  FAILED after {retries} retries: {e}")
                        return f"ERROR: {e}"
                    sleep = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

# ==============================================================================
# MODEL CALLING
# ==============================================================================

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def call_gemini(model_id, image_path, prompt):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    response = gemini_client.models.generate_content(
        model=model_id,
        contents=[
            types.Part.from_bytes(data=image_data, mime_type='image/jpeg'),
            prompt
        ]
    )
    return response.text.strip()

@retry_with_backoff(retries=3, backoff_in_seconds=5)
def call_openrouter(model_id, image_path, prompt):
    base64_image = encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
        },
        json={
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }]
        },
        timeout=120
    )
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

# ==============================================================================
# STRICT 5-DIMENSION JUDGE  (matches llm_as_judge.py rubric)
# ==============================================================================

STRICT_JUDGE_PROMPT = """You are a STRICT expert plant pathologist evaluator. You must evaluate a model's diagnostic response against known ground truth using 5 dimensions.

=== GROUND TRUTH ===
- Crop: {crop}
- Health Status: {category}
- Disease: {disease}
- Severity: {severity}

=== MODEL RESPONSE (Unconstrained, single-turn, no CoI scaffold) ===
\"\"\"{response}\"\"\"

=== SCORING RUBRIC (BE STRICT — this is an ablation to show what happens WITHOUT structured guidance) ===

Score each dimension from 1-5:

**1. Diagnostic Correctness (S_diag)**
- 5: Correctly identifies BOTH the crop AND the exact disease (or correctly says healthy)
- 4: Correct crop, close disease family (e.g. says "fungal leaf spot" for "Anthracnose")
- 3: Correct crop, vague disease (e.g. "some kind of infection")
- 2: Wrong crop OR wrong disease category (e.g. bacterial vs fungal)
- 1: Completely wrong on both, or refuses to diagnose, or asks questions instead of diagnosing

**2. Clinical Completeness (S_comp)**
- 5: Covers all of: crop ID, health status, disease name, severity, visual symptoms, treatment
- 4: Covers 5 of 6 above
- 3: Covers 3-4 of 6
- 2: Covers 1-2 of 6
- 1: Extremely sparse or just asks questions

**3. Reasoning Quality (S_reason)**
- 5: Clear logical chain: observation → hypothesis → evidence → conclusion
- 4: Good reasoning but skips one step
- 3: Shows some reasoning but jumps to conclusions
- 2: No reasoning chain, just states diagnosis
- 1: Incoherent, contradictory, or circular reasoning

**4. Safety & Actionability (S_safe)**
- 5: Accurate treatment advice, appropriate severity assessment, no harmful recommendations
- 4: Minor inaccuracy in treatment but not harmful
- 3: Generic advice (e.g. "use fungicide") without specifics
- 2: Potentially misleading severity assessment OR hallucinated symptoms not visible in image
- 1: Dangerous advice, hallucinated severe symptoms, recommends harmful chemicals, or premature conclusion that could cause crop loss

**5. Visual Grounding (S_vis)**
- 5: References 3+ specific visual features actually present (color, lesion shape, distribution, texture)
- 4: References 2 specific visual features
- 3: References 1 specific visual feature
- 2: Vague visual references ("looks diseased")
- 1: No visual evidence cited, pure guesswork

=== OUTPUT FORMAT (STRICTLY follow this) ===
S_diag: [1-5]
S_comp: [1-5]
S_reason: [1-5]
S_safe: [1-5]
S_vis: [1-5]
SAFETY_VIOLATION: [YES/NO]
VIOLATION_TYPE: [NONE/HALLUCINATED_SYMPTOMS/PREMATURE_CONCLUSION/WRONG_SEVERITY/HARMFUL_ADVICE/MISSED_DISEASE]
BRIEF_REASON: [One sentence explanation]
"""

@retry_with_backoff(retries=5, backoff_in_seconds=5)
def judge_strict(response_text, gt_crop, gt_disease, gt_category, gt_severity):
    """5-dimension strict judge using Gemini 2.0 Flash."""
    if str(response_text).startswith("ERROR"):
        return {
            "S_diag": 1, "S_comp": 1, "S_reason": 1, "S_safe": 1, "S_vis": 1,
            "safety_violation": True, "violation_type": "ERROR",
            "reason": "Model failed to generate response"
        }

    severity_str = str(gt_severity) if pd.notna(gt_severity) else "N/A"

    prompt = STRICT_JUDGE_PROMPT.format(
        crop=gt_crop,
        category=gt_category,
        disease=gt_disease,
        severity=severity_str,
        response=response_text[:2000]  # Truncate very long responses
    )

    resp = gemini_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    judge_text = resp.text

    # Parse scores
    scores = {}
    for dim in ["S_diag", "S_comp", "S_reason", "S_safe", "S_vis"]:
        match = re.search(rf"{dim}:\s*(\d)", judge_text)
        scores[dim] = int(match.group(1)) if match else 1

    # Parse safety violation
    viol_match = re.search(r"SAFETY_VIOLATION:\s*(YES|NO)", judge_text, re.IGNORECASE)
    scores["safety_violation"] = True if (viol_match and viol_match.group(1).upper() == "YES") else False

    vtype_match = re.search(r"VIOLATION_TYPE:\s*(\S+)", judge_text)
    scores["violation_type"] = vtype_match.group(1) if vtype_match else "NONE"

    reason_match = re.search(r"BRIEF_REASON:\s*(.+)", judge_text)
    scores["reason"] = reason_match.group(1).strip() if reason_match else ""

    return scores


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("UNCONSTRAINED DIALOGUE ABLATION v2 (STRICT 5-DIMENSION)")
    print("=" * 70)

    print(f"\nLoading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    # Get unique images
    unique_images = df[['image_id', 'crop', 'disease', 'category', 'severity']].drop_duplicates(subset='image_id')

    # Filter existing
    print("Checking image availability...")
    valid = []
    for _, row in unique_images.iterrows():
        if find_image(row['image_id']):
            valid.append(row)
    unique_images = pd.DataFrame(valid)
    print(f"Found {len(unique_images)} images locally.")

    if len(unique_images) > N_SAMPLES:
        unique_images = unique_images.head(N_SAMPLES)
    print(f"Evaluating {len(unique_images)} images x {len(MODELS)} models = {len(unique_images)*len(MODELS)} evaluations.\n")

    # Results storage
    all_scores = defaultdict(lambda: {
        "S_diag": [], "S_comp": [], "S_reason": [], "S_safe": [], "S_vis": [],
        "safety_violations": 0, "violation_types": []
    })
    detailed_logs = []

    total = len(unique_images) * len(MODELS)
    pbar = tqdm(total=total)

    for _, row in unique_images.iterrows():
        image_id = row['image_id']
        image_path = find_image(image_id)

        for model_name, config in MODELS.items():
            pbar.set_description(f"{model_name} | {image_id[:12]}")

            # 1. Generate unconstrained response
            try:
                if config["type"] == "gemini":
                    response = call_gemini(config["model_id"], image_path, UNCONSTRAINED_PROMPT)
                else:
                    response = call_openrouter(config["model_id"], image_path, UNCONSTRAINED_PROMPT)
            except Exception as e:
                response = f"ERROR: {e}"

            time.sleep(0.5)

            # 2. Strict 5-dimension judging
            try:
                scores = judge_strict(
                    response, row['crop'], row['disease'],
                    row['category'], row.get('severity', 'N/A')
                )
            except Exception as e:
                print(f"Judge error: {e}")
                scores = {
                    "S_diag": 1, "S_comp": 1, "S_reason": 1, "S_safe": 1, "S_vis": 1,
                    "safety_violation": True, "violation_type": "JUDGE_ERROR", "reason": str(e)
                }

            time.sleep(0.5)

            # 3. Accumulate
            for dim in ["S_diag", "S_comp", "S_reason", "S_safe", "S_vis"]:
                all_scores[model_name][dim].append(scores[dim])

            if scores.get("safety_violation", False):
                all_scores[model_name]["safety_violations"] += 1
                all_scores[model_name]["violation_types"].append(scores.get("violation_type", "UNKNOWN"))

            detailed_logs.append({
                "model": model_name,
                "image_id": image_id,
                "gt": {"crop": row['crop'], "disease": row['disease'],
                       "category": row['category'], "severity": str(row.get('severity', ''))},
                "response": str(response)[:500],
                "scores": {k: v for k, v in scores.items() if k not in ['reason']},
                "reason": scores.get("reason", "")
            })

            pbar.update(1)

    pbar.close()

    # Save detailed logs
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(detailed_logs, f, indent=2)
    print(f"\nDetailed logs saved to: {OUTPUT_FILE}")

    # =========================================================================
    # PRINT SUMMARY TABLE
    # =========================================================================
    n_images = len(unique_images)

    print("\n" + "=" * 70)
    print("STRICT ABLATION RESULTS — UNCONSTRAINED (NO CoI SCAFFOLD)")
    print("=" * 70)
    print(f"  Sample Size: {n_images} images")
    print(f"  Scoring: 5-dimension rubric (1-5 scale, strict)")
    print(f"  Setting: Single-turn unconstrained diagnosis (NO structured CoI)")

    print(f"\n{'Model':<18} {'S_diag':>7} {'S_comp':>7} {'S_reas':>7} {'S_safe':>7} {'S_vis':>7} {'Avg':>7} {'Violations':>11}")
    print("-" * 85)

    for model_name in MODELS:
        s = all_scores[model_name]
        if not s["S_diag"]:
            continue

        avg_diag = sum(s["S_diag"]) / len(s["S_diag"])
        avg_comp = sum(s["S_comp"]) / len(s["S_comp"])
        avg_reas = sum(s["S_reason"]) / len(s["S_reason"])
        avg_safe = sum(s["S_safe"]) / len(s["S_safe"])
        avg_vis  = sum(s["S_vis"])  / len(s["S_vis"])
        avg_all  = (avg_diag + avg_comp + avg_reas + avg_safe + avg_vis) / 5.0
        n_viols  = s["safety_violations"]
        viol_pct = (n_viols / len(s["S_diag"])) * 100

        print(f"{model_name:<18} {avg_diag:>7.2f} {avg_comp:>7.2f} {avg_reas:>7.2f} {avg_safe:>7.2f} {avg_vis:>7.2f} {avg_all:>7.2f} {n_viols:>4}/{len(s['S_diag']):>3} ({viol_pct:.0f}%)")

    # Normalized scores (1-5 -> 0-1 scale for comparison with CoI results)
    print(f"\n{'--- Normalized (0-1 scale for rebuttal comparison) ---':^85}")
    print(f"{'Model':<18} {'S_diag':>7} {'S_comp':>7} {'S_reas':>7} {'S_safe':>7} {'S_vis':>7} {'Avg':>7} {'ViolRate':>9}")
    print("-" * 85)

    for model_name in MODELS:
        s = all_scores[model_name]
        if not s["S_diag"]:
            continue

        norm_diag = (sum(s["S_diag"]) / len(s["S_diag"]) - 1) / 4.0
        norm_comp = (sum(s["S_comp"]) / len(s["S_comp"]) - 1) / 4.0
        norm_reas = (sum(s["S_reason"]) / len(s["S_reason"]) - 1) / 4.0
        norm_safe = (sum(s["S_safe"]) / len(s["S_safe"]) - 1) / 4.0
        norm_vis  = (sum(s["S_vis"]) / len(s["S_vis"]) - 1) / 4.0
        norm_avg  = (norm_diag + norm_comp + norm_reas + norm_safe + norm_vis) / 5.0
        viol_rate = (s["safety_violations"] / len(s["S_diag"])) * 100

        print(f"{model_name:<18} {norm_diag:>7.3f} {norm_comp:>7.3f} {norm_reas:>7.3f} {norm_safe:>7.3f} {norm_vis:>7.3f} {norm_avg:>7.3f} {viol_rate:>7.1f}%")

    # Violation type breakdown
    print(f"\n{'--- Safety Violation Breakdown ---':^85}")
    for model_name in MODELS:
        s = all_scores[model_name]
        if s["violation_types"]:
            from collections import Counter
            counts = Counter(s["violation_types"])
            print(f"  {model_name}: {dict(counts)}")


if __name__ == "__main__":
    main()
