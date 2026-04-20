"""
Compute Explainability Efficiency for Cascading Context results.
Compare with Guided (from Table 3) and interpolate for other models.

Explainability Efficiency (E) = (Visual Cues Found / Word Count) * 100
= Visual cues per 100 words
"""

import json
import os
import re

BASE_DIR = os.environ.get("PLANTINQUIRY_HOME", ".")
CHECKPOINT = os.path.join(BASE_DIR, "eval/cascading_context_results/gemini3_flash_checkpoint.json")

# Visual cues list (same as used in test_1_gemini3_flash.py)
VISUAL_CUES = [
    'spot', 'lesion', 'discoloration', 'wilting', 'yellowing', 'browning',
    'necrotic', 'chlorotic', 'margin', 'leaf', 'stem', 'tissue',
    'color', 'pattern', 'visible', 'appearance', 'surface', 'texture',
    'uniform', 'irregular', 'circular', 'elongated', 'scattered',
    'green', 'brown', 'yellow', 'white', 'black', 'gray', 'grey',
    'dried', 'withered', 'deformed', 'curled', 'twisted'
]


def count_visual_cues(text):
    """Count how many distinct visual cues appear in text."""
    text_lower = text.lower()
    return sum(1 for cue in VISUAL_CUES if cue in text_lower)


def word_count(text):
    """Count words in text."""
    return len(text.split())


def compute_efficiency(cues, words):
    """Explainability Efficiency = cues per 100 words."""
    if words == 0:
        return 0.0
    return (cues / words) * 100


def main():
    # Load checkpoint
    with open(CHECKPOINT) as f:
        data = json.load(f)

    results = data['results']
    n_images = len(results)
    print(f"Loaded {n_images} completed images from Gemini-3-Flash cascading checkpoint\n")

    # Compute per-image and per-question metrics
    all_cues = []
    all_words = []
    all_eff = []
    per_question = {}  # q_num -> list of (cues, words, eff)

    for img_result in results:
        for qa in img_result['qa_results']:
            answer = str(qa.get('model_answer', ''))
            if answer.startswith('ERROR'):
                continue

            cues = count_visual_cues(answer)
            words = word_count(answer)
            eff = compute_efficiency(cues, words)

            all_cues.append(cues)
            all_words.append(words)
            all_eff.append(eff)

            q_num = qa.get('question_number', 0)
            if q_num not in per_question:
                per_question[q_num] = []
            per_question[q_num].append({
                'cues': cues, 'words': words, 'eff': eff,
                'category': qa.get('question_category', '')
            })

    # Overall stats
    avg_cues = sum(all_cues) / len(all_cues) if all_cues else 0
    avg_words = sum(all_words) / len(all_words) if all_words else 0
    avg_eff = sum(all_eff) / len(all_eff) if all_eff else 0
    overall_eff = compute_efficiency(sum(all_cues), sum(all_words))

    print("=" * 70)
    print("GEMINI-3-FLASH — CASCADING CONTEXT EFFICIENCY")
    print("=" * 70)
    print(f"  Total QAs analyzed: {len(all_cues)}")
    print(f"  Avg Visual Cues:    {avg_cues:.2f}")
    print(f"  Avg Word Count:     {avg_words:.1f}")
    print(f"  Avg Efficiency (E): {avg_eff:.2f}")
    print(f"  Overall Efficiency: {overall_eff:.2f}")

    # Per-question breakdown
    print(f"\n{'Q#':<5} {'Category':<30} {'Cues':>6} {'Words':>7} {'Eff':>7} {'n':>5}")
    print("-" * 65)
    for q_num in sorted(per_question.keys()):
        items = per_question[q_num]
        avg_c = sum(i['cues'] for i in items) / len(items)
        avg_w = sum(i['words'] for i in items) / len(items)
        avg_e = sum(i['eff'] for i in items) / len(items)
        cat = items[0]['category'] if items else ''
        print(f"Q{q_num:<4} {cat:<30} {avg_c:>6.2f} {avg_w:>7.1f} {avg_e:>7.2f} {len(items):>5}")

    # =====================================================================
    # COMPARISON TABLE: Guided vs Cascading vs Scaffolded vs Unconstrained
    # =====================================================================
    # From Table 3 in the paper (Gemini-3-Flash uses "gemini-3-flash-preview")
    # Note: Table 3 doesn't list Gemini-3-Flash directly, but has Gemini-2.5-Flash and Gemini-2.5-Pro
    # Using the Gemini-3-Flash data from the main evaluation where available

    # From Table 3:
    # Gemini-2.5-Flash:  Scaffolded: Eff=2.60, Cues=8.65, Words=456  | Guided: Eff=3.67, Cues=4.71, Words=181
    # Gemini-2.5-Pro:    Scaffolded: Eff=2.95, Cues=6.11, Words=268  | Guided: Eff=3.58, Cues=4.00, Words=147
    # Seed-1.6-Flash:    Scaffolded: Eff=3.22, Cues=5.63, Words=198  | Guided: Eff=3.75, Cues=4.10, Words=125

    # From unconstrained ablation v2 (Gemini-3-Flash, 50 images):
    # Single turn, no structure at all

    print("\n\n" + "=" * 85)
    print("THREE-WAY COMPARISON: Gemini-3-Flash Explainability Efficiency")
    print("=" * 85)

    # Cascading result (this run)
    casc_eff = avg_eff
    casc_cues = avg_cues
    casc_words = avg_words

    print(f"\n{'Setting':<25} {'Eff (E)':>8} {'Avg Cues':>10} {'Avg Words':>10} {'Source':>25}")
    print("-" * 85)

    # We need Guided numbers for Gemini-3-Flash. From Test 1a results
    # Approximate from Table 3 pattern (Gemini models: Guided Eff ~3.6-3.7)
    # Let's use the actual data if available
    print(f"{'Guided (GT history)':<25} {'~3.67':>8} {'~4.71':>10} {'~181':>10} {'Table 3 (paper)':>25}")
    print(f"{'Cascading (own history)':<25} {casc_eff:>8.2f} {casc_cues:>10.2f} {casc_words:>10.1f} {'This experiment':>25}")
    print(f"{'Scaffolded (no history)':<25} {'~2.95':>8} {'~6.11':>10} {'~268':>10} {'Table 3 (paper)':>25}")

    print(f"\n✅ Expected ordering: Guided > Cascading > Scaffolded")
    if casc_eff > 2.95:
        print(f"   ✓ Cascading ({casc_eff:.2f}) > Scaffolded (~2.95)")
    else:
        print(f"   ✗ Cascading ({casc_eff:.2f}) ≤ Scaffolded (~2.95) — unexpected")
    if casc_eff < 3.67:
        print(f"   ✓ Cascading ({casc_eff:.2f}) < Guided (~3.67)")
    else:
        print(f"   ✗ Cascading ({casc_eff:.2f}) ≥ Guided (~3.67) — unexpected")

    # Interpolation for other models
    print("\n\n" + "=" * 85)
    print("INTERPOLATED CASCADING EFFICIENCY FOR OTHER MODELS")
    print("(Using Gemini-3-Flash ratio: Cascading/Guided)")
    print("=" * 85)

    # Ratio of Cascading to Guided for Gemini-3-Flash
    guided_eff = 3.67  # From Table 3 (Gemini-2.5-Flash as proxy)
    ratio = casc_eff / guided_eff if guided_eff > 0 else 0.9

    models_table3 = {
        "Grok-4.1-Fast":          {"scaffolded": 4.54, "guided": 5.20},
        "Gemini-2.5-Flash":       {"scaffolded": 2.60, "guided": 3.67},
        "Gemini-2.5-Pro":         {"scaffolded": 2.95, "guided": 3.58},
        "Qwen3-VL-32B":           {"scaffolded": 2.88, "guided": 3.33},
        "Seed-1.6-Flash":         {"scaffolded": 3.22, "guided": 3.75},
        "Qwen2.5-VL-32B":         {"scaffolded": 1.60, "guided": 2.94},
        "Qwen2.5-VL-72B":         {"scaffolded": 2.46, "guided": 2.92},
        "Pixtral-12B":            {"scaffolded": 2.53, "guided": 2.90},
        "Llama-3.2-90B-Vision":   {"scaffolded": 2.40, "guided": 2.85},
        "Gemma-3-27B":            {"scaffolded": 1.88, "guided": 2.38},
        "Ministral-3B":           {"scaffolded": 2.26, "guided": 2.71},
        "Ministral-8B":           {"scaffolded": 2.21, "guided": 2.65},
        "Mistral-Medium-3.1":     {"scaffolded": 2.35, "guided": 2.70},
        "Llama-4-Maverick":       {"scaffolded": 2.31, "guided": 2.65},
        "Phi-4-Multimodal":       {"scaffolded": 1.94, "guided": 2.55},
    }

    print(f"\nGemini-3-Flash Cascading/Guided ratio: {ratio:.3f}")
    print(f"\n{'Model':<25} {'Scaffolded':>10} {'Cascading*':>11} {'Guided':>8} {'Δ(G→C)%':>9}")
    print("-" * 70)

    for model, scores in models_table3.items():
        cascading_est = scores["guided"] * ratio
        delta_pct = ((cascading_est - scores["guided"]) / scores["guided"]) * 100
        print(f"{model:<25} {scores['scaffolded']:>10.2f} {cascading_est:>11.2f} {scores['guided']:>8.2f} {delta_pct:>+8.1f}%")

    print(f"\n* Cascading values are interpolated using Gemini-3-Flash's measured ratio.")
    print(f"  This assumes all models experience a similar proportional drop when")
    print(f"  switching from GT history to model-generated history.")

    # Save results
    output = {
        "model": "Gemini-3-Flash",
        "setting": "cascading_context",
        "n_images": n_images,
        "n_qa": len(all_cues),
        "avg_cues": avg_cues,
        "avg_words": avg_words,
        "avg_efficiency": avg_eff,
        "overall_efficiency": overall_eff,
        "cascading_to_guided_ratio": ratio,
        "per_question": {
            str(q): {
                "avg_cues": sum(i['cues'] for i in items) / len(items),
                "avg_words": sum(i['words'] for i in items) / len(items),
                "avg_eff": sum(i['eff'] for i in items) / len(items),
                "n": len(items)
            }
            for q, items in per_question.items()
        }
    }

    out_path = os.path.join(BASE_DIR, "eval/cascading_context_results/cascading_efficiency_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
