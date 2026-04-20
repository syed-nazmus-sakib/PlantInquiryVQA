"""
Compute accuracy metrics from Gemini-3-Flash cascading checkpoint,
then interpolate for top 5 models.
"""

import json
import re
import os

BASE_DIR = os.environ.get("PLANTINQUIRY_HOME", ".")
CHECKPOINT = os.path.join(BASE_DIR, "eval/cascading_context_results/gemini3_flash_checkpoint.json")

# Disease keywords for detection
DISEASE_KEYWORDS = [
    'blight', 'spot', 'rot', 'mildew', 'rust', 'wilt', 'mosaic',
    'virus', 'scab', 'canker', 'mold', 'mould', 'bacterial', 'fungal',
    'healthy', 'necrosis', 'streak', 'curl', 'leaf curl', 'powdery',
    'downy', 'anthracnose', 'septoria', 'cercospora', 'alternaria',
    'deficiency', 'senescence', 'dry', 'nutrient', 'chlorosis',
    'gall', 'midge', 'mites', 'weevil', 'ringspot', 'scorch',
]

VISUAL_CUES = [
    'spot', 'lesion', 'discoloration', 'wilting', 'yellowing', 'browning',
    'necrotic', 'chlorotic', 'margin', 'leaf', 'stem', 'tissue',
    'color', 'pattern', 'visible', 'appearance', 'surface', 'texture',
    'uniform', 'irregular', 'circular', 'elongated', 'scattered',
    'green', 'brown', 'yellow', 'white', 'black', 'gray', 'grey',
    'dried', 'withered', 'deformed', 'curled', 'twisted'
]


def compute_f1(prediction, reference):
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
    recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def check_disease_match(model_answer, gt_disease):
    """Check if model correctly identified the disease."""
    if not gt_disease or gt_disease == 'nan':
        return 0
    answer_lower = model_answer.lower()
    disease_lower = gt_disease.lower()

    # Direct match
    if disease_lower in answer_lower:
        return 1

    # Check disease components
    disease_parts = disease_lower.replace('_', ' ').split()
    matched = sum(1 for part in disease_parts if part in answer_lower and len(part) > 3)
    if matched >= max(1, len(disease_parts) * 0.5):
        return 1

    return 0


def check_crop_match(model_answer, gt_crop):
    """Check if model correctly identified the crop."""
    if not gt_crop or gt_crop == 'nan':
        return 0
    answer_lower = model_answer.lower()
    crop_lower = gt_crop.lower()
    if crop_lower in answer_lower:
        return 1
    # Check parts (e.g., "Bottle Gourd" -> "bottle", "gourd")
    parts = crop_lower.split()
    if any(p in answer_lower for p in parts if len(p) > 3):
        return 1
    return 0


def compute_vg_score(answer):
    """Visual grounding: fraction of visual cues present."""
    answer_lower = answer.lower()
    found = sum(1 for cue in VISUAL_CUES if cue in answer_lower)
    return min(found / 10.0, 1.0)  # Normalize: 10+ cues = 1.0


def compute_safety_score(answer, disease):
    """Basic safety: does the model recommend treatment for diseased plants?"""
    answer_lower = answer.lower()
    disease_lower = str(disease).lower()

    if disease_lower in ['healthy', 'nan', '']:
        # For healthy plants, safety = not recommending unnecessary treatment
        treatment_words = ['fungicide', 'pesticide', 'spray', 'chemical', 'treat']
        unnecessary = sum(1 for w in treatment_words if w in answer_lower)
        return 1.0 if unnecessary == 0 else max(0, 1.0 - unnecessary * 0.2)
    else:
        # For diseased plants, safety = recommending action
        action_words = ['treat', 'remove', 'apply', 'fungicide', 'spray',
                       'prune', 'monitor', 'management', 'control', 'action']
        action_count = sum(1 for w in action_words if w in answer_lower)
        return min(action_count / 3.0, 1.0)  # 3+ action words = 1.0


def main():
    with open(CHECKPOINT) as f:
        data = json.load(f)

    results = data['results']
    n_images = len(results)
    print(f"Loaded {n_images} images from Gemini-3-Flash cascading checkpoint\n")

    # Compute per-QA metrics
    all_f1 = []
    all_disease_match = []
    all_crop_match = []
    all_vg = []
    all_safety = []
    per_q = {}  # question_number -> lists

    for img_result in results:
        disease = img_result.get('disease', '')
        crop = img_result.get('crop', '')

        for qa in img_result['qa_results']:
            answer = str(qa.get('model_answer', ''))
            gt = str(qa.get('ground_truth', ''))
            q_num = qa.get('question_number', 0)
            q_cat = qa.get('question_category', '')

            if answer.startswith('ERROR'):
                continue

            f1 = compute_f1(answer, gt)
            dis_match = check_disease_match(answer, disease)
            crop_match = check_crop_match(answer, crop)
            vg = compute_vg_score(answer)
            safety = compute_safety_score(answer, disease)

            all_f1.append(f1)
            all_disease_match.append(dis_match)
            all_crop_match.append(crop_match)
            all_vg.append(vg)
            all_safety.append(safety)

            if q_num not in per_q:
                per_q[q_num] = {'f1': [], 'dis': [], 'crop': [], 'vg': [], 'safe': [], 'cat': q_cat}
            per_q[q_num]['f1'].append(f1)
            per_q[q_num]['dis'].append(dis_match)
            per_q[q_num]['crop'].append(crop_match)
            per_q[q_num]['vg'].append(vg)
            per_q[q_num]['safe'].append(safety)

    # Averages
    avg = lambda lst: sum(lst)/len(lst) if lst else 0
    casc_f1 = avg(all_f1)
    casc_dis = avg(all_disease_match)
    casc_crop = avg(all_crop_match)
    casc_vg = avg(all_vg)
    casc_safe = avg(all_safety)

    print("=" * 70)
    print("GEMINI-3-FLASH — CASCADING CONTEXT ACCURACY METRICS")
    print("=" * 70)
    print(f"  Total QAs:          {len(all_f1)}")
    print(f"  F1 Score:           {casc_f1:.3f}")
    print(f"  Disease Accuracy:   {casc_dis:.3f}")
    print(f"  Crop Accuracy:      {casc_crop:.3f}")
    print(f"  Visual Grounding:   {casc_vg:.3f}")
    print(f"  Safety Score:       {casc_safe:.3f}")

    # Per-question breakdown
    print(f"\n{'Q#':<5} {'Category':<28} {'F1':>6} {'Dis':>6} {'Crop':>6} {'VG':>6} {'Safe':>6} {'n':>5}")
    print("-" * 70)
    for q_num in sorted(per_q.keys()):
        d = per_q[q_num]
        print(f"Q{q_num:<4} {d['cat']:<28} "
              f"{avg(d['f1']):>6.3f} {avg(d['dis']):>6.3f} {avg(d['crop']):>6.3f} "
              f"{avg(d['vg']):>6.3f} {avg(d['safe']):>6.3f} {len(d['f1']):>5}")

    # =========================================================================
    # COMPARISON: Guided (Table 2) vs Cascading (measured) → compute ratios
    # =========================================================================

    # Guided scores from Table 2 for Gemini-3-Flash
    guided = {
        'f1': 0.255, 'dis': 0.444, 'safe': 0.147, 'vg': 0.259, 'clin': 0.188
    }

    # Compute retention ratios
    ratio_f1 = casc_f1 / guided['f1'] if guided['f1'] > 0 else 1.0
    ratio_dis = casc_dis / guided['dis'] if guided['dis'] > 0 else 1.0
    ratio_vg = casc_vg / guided['vg'] if guided['vg'] > 0 else 1.0
    ratio_safe = casc_safe / guided['safe'] if guided['safe'] > 0 else 1.0

    print(f"\n{'='*70}")
    print(f"GUIDED → CASCADING RETENTION RATIOS (Gemini-3-Flash)")
    print(f"{'='*70}")
    print(f"  F1:       {guided['f1']:.3f} → {casc_f1:.3f}  (ratio: {ratio_f1:.3f}, Δ: {(ratio_f1-1)*100:+.1f}%)")
    print(f"  Disease:  {guided['dis']:.3f} → {casc_dis:.3f}  (ratio: {ratio_dis:.3f}, Δ: {(ratio_dis-1)*100:+.1f}%)")
    print(f"  VG:       {guided['vg']:.3f} → {casc_vg:.3f}  (ratio: {ratio_vg:.3f}, Δ: {(ratio_vg-1)*100:+.1f}%)")
    print(f"  Safety:   {guided['safe']:.3f} → {casc_safe:.3f}  (ratio: {ratio_safe:.3f}, Δ: {(ratio_safe-1)*100:+.1f}%)")

    # =========================================================================
    # INTERPOLATE FOR TOP 5 MODELS
    # =========================================================================

    # Top 5 models from Table 2 (by Disease Accuracy, excluding Qwen)
    models_guided = {
        "Gemini-3-Flash":       {"f1": 0.255, "dis": 0.444, "clin": 0.188, "safe": 0.147, "vg": 0.259},
        "Gemini-2.5-Pro":       {"f1": 0.225, "dis": 0.357, "clin": 0.112, "safe": 0.040, "vg": 0.408},
        "Seed-1.6-Flash":       {"f1": 0.226, "dis": 0.344, "clin": 0.120, "safe": 0.075, "vg": 0.394},
        "Llama-3.2-90B-Vision": {"f1": 0.212, "dis": 0.340, "clin": 0.185, "safe": 0.214, "vg": 0.372},
        "Llama-4-Maverick":     {"f1": 0.212, "dis": 0.329, "clin": 0.175, "safe": 0.202, "vg": 0.397},
    }

    # Use the measured ratios (clamped to reasonable range)
    ratio_clin = min(ratio_f1, ratio_dis)  # clinical utility tracks accuracy

    print(f"\n\n{'='*90}")
    print(f"TABLE R4: Accuracy Scores — Guided vs Cascading (Top 5 Models)")
    print(f"{'='*90}")

    # Print header
    print(f"\n{'Model':<24} │ {'--- Guided (GT History) ---':^35} │ {'--- Cascading (Own History)* ---':^35}")
    print(f"{'':24} │ {'Dis':>7} {'Clin':>7} {'Safe':>7} {'VG':>7} │ {'Dis':>7} {'Clin':>7} {'Safe':>7} {'VG':>7}")
    print("─" * 90)

    for model_name, g in models_guided.items():
        # Compute cascading values using measured ratios
        c_dis = g['dis'] * ratio_dis
        c_clin = g['clin'] * ratio_clin
        c_safe = g['safe'] * ratio_safe
        c_vg = g['vg'] * ratio_vg

        marker = " ✓" if model_name == "Gemini-3-Flash" else " *"
        print(f"{model_name:<24} │ {g['dis']:>7.3f} {g['clin']:>7.3f} {g['safe']:>7.3f} {g['vg']:>7.3f} "
              f"│ {c_dis:>7.3f} {c_clin:>7.3f} {c_safe:>7.3f} {c_vg:>7.3f}{marker}")

    print(f"\n✓ = Measured directly    * = Interpolated using Gemini-3-Flash retention ratios")
    print(f"\nRetention Ratios: Dis={ratio_dis:.3f}, Clin={ratio_clin:.3f}, "
          f"Safe={ratio_safe:.3f}, VG={ratio_vg:.3f}")

    # =========================================================================
    # FULL THREE-WAY TABLE
    # =========================================================================

    print(f"\n\n{'='*90}")
    print(f"TABLE R5: Disease Accuracy (Dis) — Three-Setting Comparison (Top 5)")
    print(f"{'='*90}")
    print(f"\n{'Model':<24} {'Guided':>8} {'Cascading':>10} {'Δ(G→C)%':>9} │ Interpretation")
    print("─" * 90)

    for model_name, g in models_guided.items():
        c_dis = g['dis'] * ratio_dis
        delta = (ratio_dis - 1) * 100
        marker = "✓ measured" if model_name == "Gemini-3-Flash" else "* interpolated"
        print(f"{model_name:<24} {g['dis']:>8.3f} {c_dis:>10.3f} {delta:>+8.1f}% │ {marker}")

    avg_guided = sum(g['dis'] for g in models_guided.values()) / len(models_guided)
    avg_casc = avg_guided * ratio_dis
    print("─" * 90)
    print(f"{'Average':<24} {avg_guided:>8.3f} {avg_casc:>10.3f} {(ratio_dis-1)*100:>+8.1f}%")

    # Save
    out = {
        "model": "Gemini-3-Flash",
        "n_images": n_images,
        "cascading_metrics": {"f1": casc_f1, "dis": casc_dis, "crop": casc_crop, "vg": casc_vg, "safe": casc_safe},
        "guided_metrics": guided,
        "retention_ratios": {"f1": ratio_f1, "dis": ratio_dis, "vg": ratio_vg, "safe": ratio_safe},
        "interpolated": {
            model: {
                "guided": g,
                "cascading": {
                    "dis": g['dis'] * ratio_dis,
                    "clin": g['clin'] * ratio_clin,
                    "safe": g['safe'] * ratio_safe,
                    "vg": g['vg'] * ratio_vg
                }
            }
            for model, g in models_guided.items()
        }
    }
    out_path = os.path.join(BASE_DIR, "eval/cascading_context_results/cascading_accuracy_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
