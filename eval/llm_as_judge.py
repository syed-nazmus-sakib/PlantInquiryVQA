#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation for PlantInquiryVQA
============================================
Uses an LLM (via OpenRouter) to score model answers against ground truth
on 5 dimensions, replacing keyword-matching metrics with semantic evaluation.

Addresses ACL reviewer concerns about:
- Keyword overlap under-counting synonyms (R1-W1, R2-W2)
- Arbitrary composite weighting α,β,γ (R2-W2)  
- VLM vocabulary bias in grounding scores (R1-Q2)
"""

import os
import json
import time
import random
import csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import requests
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# Configuration
# ==============================================================================
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Judge model
JUDGE_MODEL = "google/gemini-3-flash-preview"

# Paths to saved results
RESULTS_BASE = os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "eval")

# Models to evaluate (path_relative_to_RESULTS_BASE -> display_name)
# ALL 17+ models from Table 2 of the paper for comprehensive ACL rebuttal
# Uses test1a (conversational CoI) where available, test2 for additional models
MODELS_TO_EVALUATE = {
    # =========================================================================
    # Gemini family (test1a - conversational chain-of-inquiry)
    # =========================================================================
    "test1a_gemini_results/gemini3_Flash/test1a_gemini3_results.json": "Gemini-3-Flash",
    "test1a_gemini_results/gemini25_pro/results.json": "Gemini-2.5-Pro",
    "test1a_gemini_results/gemini25_flash/results.json": "Gemini-2.5-Flash",
    "test1a_gemini_results/gemma_3_27b_results/results.json": "Gemma-3-27B",

    # =========================================================================
    # Qwen family (test1a)
    # =========================================================================
    "test1a_qwen_results/qwen3_vl_235b/results.json": "Qwen3-VL-235B",
    "test1a_qwen_results/qwen3_vl_32b/results.json": "Qwen3-VL-32B",
    "test1a_qwen_results/qwen25_vl_72b/results.json": "Qwen2.5-VL-72B",
    "test1a_qwen_results/qwen25_vl_32b/results.json": "Qwen2.5-VL-32B",
    "test1a_qwen_results/qwen_vl_plus/results.json": "Qwen-VL-Plus",

    # =========================================================================
    # Ministral family (test1a)
    # =========================================================================
    "test1a_ministral_results/ministral_8b/results.json": "Ministral-8B",
    "test1a_ministral_results/ministral_3b/results.json": "Ministral-3B",

    # =========================================================================
    # Additional models (test2 - question scaffolding format)
    # These models only have test2 results available
    # =========================================================================
    "test2_additional_models_results/seed_1_6_flash/results.json": "Seed-1.6-Flash",
    "test2_additional_models_results/llama_3_2_90b_vision/results.json": "Llama-3.2-90B-Vision",
    "test2_additional_models_results/llama_4_maverick/results.json": "Llama-4-Maverick",
    "test2_additional_models_results/pixtral_12b/results.json": "Pixtral-12B",
    "test2_additional_models_results/phi_4_multimodal/results.json": "Phi-4-Multimodal",
    "test2_additional_models_results/grok_4_1_fast/results.json": "Grok-4.1-Fast",
    "test2_additional_models_results/mistral_medium_3_1/results.json": "Mistral-Medium-3.1",
}

# How many images to sample for the representative subset
# Increased to 50 for stronger statistical power in ACL rebuttal
N_SAMPLE_IMAGES = 50

# Output
OUTPUT_DIR = os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "eval/llm_judge_results")

# Rate limiting
MIN_DELAY_SECONDS = 0.1  # Between API calls (API isn't rate limiting us)
SAVE_INTERVAL = 20  # Save checkpoint every N judgments

# ==============================================================================
# Judge Prompt & Rubric
# ==============================================================================
JUDGE_SYSTEM_PROMPT = """You are an expert plant pathologist evaluating AI model responses about plant diseases. 
You will be given a question about a plant image, the ground truth answer, and a model's predicted answer.
Score the model's answer on 5 dimensions using a 1-5 scale.

SCORING RUBRIC:

1. DIAGNOSTIC CORRECTNESS (1-5)
   1 = Completely wrong disease/crop identification
   2 = Partially correct but with major errors (wrong disease family)
   3 = Correct general direction but imprecise (e.g., says "fungal disease" but not specific name)
   4 = Correct identification, minor details off
   5 = Perfectly correct, including synonyms/alternative names
   NOTE: Accept valid synonyms! e.g., "Alternaria leaf spot" = "early blight", "Colletotrichum" = "anthracnose"

2. CLINICAL COMPLETENESS (1-5)
   1 = Missing almost all relevant clinical information
   2 = Mentions only 1-2 relevant points
   3 = Covers some key points but misses important aspects
   4 = Comprehensive coverage with minor omissions
   5 = Thorough and complete clinical information

3. REASONING QUALITY (1-5)
   1 = No reasoning, just assertions
   2 = Superficial reasoning with logical gaps
   3 = Basic reasoning present but could be stronger
   4 = Good logical flow connecting observations to conclusions
   5 = Expert-level reasoning with clear evidence-based logic

4. SAFETY & ACTIONABILITY (1-5)
   1 = Dangerous or misleading recommendations
   2 = Vague/unhelpful, could lead to wrong action
   3 = Generally safe but not very actionable
   4 = Safe and reasonably actionable recommendations
   5 = Safe, specific, and immediately actionable advice
   NOTE: For non-management questions, score based on whether the information provided is safe/not misleading

5. VISUAL GROUNDING (1-5)
   1 = No reference to visual features at all
   2 = Generic visual mentions not specific to this case
   3 = Some relevant visual descriptions
   4 = Good visual descriptions that support the diagnosis
   5 = Detailed, specific visual descriptions that clearly ground the diagnosis

RESPOND IN EXACTLY THIS JSON FORMAT (no other text):
{
  "diagnostic_correctness": <1-5>,
  "clinical_completeness": <1-5>,
  "reasoning_quality": <1-5>,
  "safety_actionability": <1-5>,
  "visual_grounding": <1-5>,
  "brief_justification": "<one sentence explaining the overall quality>"
}"""


def build_judge_prompt(question: str, ground_truth: str, model_answer: str,
                       crop: str, disease: str, severity: str,
                       question_category: str) -> str:
    """Build the user prompt for the judge."""
    return f"""Evaluate this plant disease VQA response:

**Context:**
- Crop: {crop}
- Disease: {disease}
- Severity: {severity}
- Question Category: {question_category}

**Question:** {question}

**Ground Truth Answer:** {ground_truth}

**Model's Answer:** {model_answer}

Score the model's answer on the 5 dimensions (1-5 each) following the rubric. Return ONLY the JSON."""


# ==============================================================================
# API Calls
# ==============================================================================
def call_judge(prompt: str, max_retries: int = 3) -> Optional[Dict]:
    """Call the judge LLM via OpenRouter and parse the response."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url=OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://plantinquiryvqa.github.io",
                    "X-Title": "PlantInquiryVQA Judge",
                },
                data=json.dumps({
                    "model": JUDGE_MODEL,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 300,
                }),
                timeout=60
            )
            
            if response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            
            if response.status_code != 200:
                print(f"  API error {response.status_code}: {response.text[:200]}")
                time.sleep(5)
                continue
            
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Parse JSON from content (handle potential markdown wrapping)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            scores = json.loads(content)
            
            # Validate scores
            required_keys = ["diagnostic_correctness", "clinical_completeness",
                           "reasoning_quality", "safety_actionability", "visual_grounding"]
            for key in required_keys:
                if key not in scores:
                    raise ValueError(f"Missing key: {key}")
                if not isinstance(scores[key], (int, float)) or scores[key] < 1 or scores[key] > 5:
                    raise ValueError(f"Invalid score for {key}: {scores[key]}")
            
            return scores
            
        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            print(f"  Raw content: {content[:200]}")
            time.sleep(2)
        except Exception as e:
            print(f"  Error (attempt {attempt+1}): {e}")
            time.sleep(3)
    
    return None


# ==============================================================================
# Data Loading
# ==============================================================================
def load_model_results(results_path: str) -> Tuple[str, List[Dict]]:
    """Load a model's results from JSON file.
    
    Returns (model_name, list_of_image_results).
    Each image_result has: image_id, crop, disease, severity, conditions[].
    Each condition has: question, ground_truth, model_answer, question_category.
    """
    with open(results_path) as f:
        data = json.load(f)
    
    model_name = data.get('metadata', {}).get('model', 'unknown')
    
    # Handle both 'detailed_results' and 'results' keys
    results = data.get('detailed_results', data.get('results', []))
    
    # Normalize the answer field
    normalized = []
    for r in results:
        norm_r = {
            'image_id': r['image_id'],
            'crop': r.get('crop', ''),
            'disease': r.get('disease', ''),
            'severity': str(r.get('severity', '')),
            'conditions': []
        }
        
        for c in r.get('conditions', []):
            # model_answer can be top-level or nested under a model-name key
            model_answer = c.get('model_answer', '')
            if not model_answer:
                # Check for nested format: {model_key: {answer: ..., metrics: ...}}
                for k, v in c.items():
                    if isinstance(v, dict) and 'answer' in v:
                        model_answer = v['answer']
                        break
            
            norm_c = {
                'question': c.get('question', ''),
                'ground_truth': str(c.get('ground_truth', '')),
                'model_answer': str(model_answer) if model_answer else '',
                'question_category': c.get('question_category', ''),
                'question_number': c.get('question_number', 0)
            }
            norm_r['conditions'].append(norm_c)
        
        normalized.append(norm_r)
    
    return model_name, normalized


def select_representative_subset(all_results: List[Dict], n_images: int = 50) -> List[str]:
    """Select a stratified representative subset of image IDs.
    
    Stratifies by severity (MILD/MODERATE/SEVERE) and ensures crop diversity.
    """
    # Group images by severity
    by_severity = defaultdict(list)
    for r in all_results:
        sev = r['severity'].upper() if r['severity'] else 'UNKNOWN'
        by_severity[sev].append(r['image_id'])
    
    print(f"\nSeverity distribution in full results:")
    for sev, ids in sorted(by_severity.items()):
        print(f"  {sev}: {len(ids)} images")
    
    # Stratified sampling
    selected = []
    severity_counts = {
        'MILD': max(1, n_images // 4),
        'MODERATE': max(1, n_images // 3),
        'SEVERE': max(1, n_images // 4),
    }
    # Allocate remaining to whatever is available
    remaining = n_images - sum(severity_counts.values())
    
    random.seed(42)  # Reproducibility
    
    for sev, count in severity_counts.items():
        available = by_severity.get(sev, [])
        if available:
            selected.extend(random.sample(available, min(count, len(available))))
    
    # Fill remaining with any images not yet selected
    all_ids = [r['image_id'] for r in all_results]
    unselected = [i for i in all_ids if i not in set(selected)]
    if remaining > 0 and unselected:
        selected.extend(random.sample(unselected, min(remaining, len(unselected))))
    
    print(f"\nSelected {len(selected)} images for evaluation")
    return selected


# ==============================================================================
# Main Evaluation
# ==============================================================================
def evaluate_model(model_name: str, display_name: str, results: List[Dict],
                   selected_images: List[str], output_dir: str) -> Dict:
    """Evaluate a single model using LLM-as-Judge on the selected images."""
    
    model_dir = os.path.join(output_dir, display_name.replace(" ", "_").replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)
    
    # Load checkpoint if exists
    checkpoint_file = os.path.join(model_dir, "checkpoint.json")
    judged_results = []
    processed_keys = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
            judged_results = checkpoint_data.get('results', [])
            processed_keys = {(r['image_id'], r['question_number']) for r in judged_results}
            print(f"  Resuming from checkpoint: {len(judged_results)} already judged")
    
    # Filter to selected images
    selected_set = set(selected_images)
    filtered = [r for r in results if r['image_id'] in selected_set]
    
    total_to_judge = sum(len(r['conditions']) for r in filtered)
    already_done = len(processed_keys)
    print(f"  Total QA pairs to judge: {total_to_judge}, already done: {already_done}")
    
    judged_count = already_done
    
    for img_result in filtered:
        image_id = img_result['image_id']
        
        for condition in img_result['conditions']:
            key = (image_id, condition['question_number'])
            if key in processed_keys:
                continue
            
            # Skip if model answer is empty or an error
            if not condition['model_answer'] or condition['model_answer'].startswith('ERROR'):
                judged_results.append({
                    'image_id': image_id,
                    'question_number': condition['question_number'],
                    'question_category': condition['question_category'],
                    'crop': img_result['crop'],
                    'disease': img_result['disease'],
                    'severity': img_result['severity'],
                    'scores': None,
                    'skipped': True,
                    'reason': 'empty_or_error'
                })
                processed_keys.add(key)
                continue
            
            # Build judge prompt
            prompt = build_judge_prompt(
                question=condition['question'],
                ground_truth=condition['ground_truth'],
                model_answer=condition['model_answer'],
                crop=img_result['crop'],
                disease=img_result['disease'],
                severity=img_result['severity'],
                question_category=condition['question_category']
            )
            
            # Call judge
            time.sleep(MIN_DELAY_SECONDS)
            scores = call_judge(prompt)
            
            judged_results.append({
                'image_id': image_id,
                'question_number': condition['question_number'],
                'question_category': condition['question_category'],
                'crop': img_result['crop'],
                'disease': img_result['disease'],
                'severity': img_result['severity'],
                'question': condition['question'],
                'ground_truth': condition['ground_truth'][:200],
                'model_answer': condition['model_answer'][:200],
                'scores': scores,
                'skipped': scores is None,
            })
            processed_keys.add(key)
            
            judged_count += 1
            if judged_count % 5 == 0:
                print(f"  [{display_name}] Judged {judged_count}/{total_to_judge}")
            
            # Save checkpoint
            if judged_count % SAVE_INTERVAL == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'results': judged_results}, f, indent=2)
    
    # Save final results
    final_file = os.path.join(model_dir, "judge_results.json")
    with open(final_file, 'w') as f:
        json.dump({
            'model': display_name,
            'judge_model': JUDGE_MODEL,
            'n_images': len(filtered),
            'n_judged': len([r for r in judged_results if not r.get('skipped')]),
            'timestamp': datetime.now().isoformat(),
            'results': judged_results
        }, f, indent=2)
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    print(f"  [{display_name}] Done! Saved to {final_file}")
    return aggregate_model_scores(display_name, judged_results)


def aggregate_model_scores(display_name: str, judged_results: List[Dict]) -> Dict:
    """Aggregate scores for a single model."""
    dimensions = ["diagnostic_correctness", "clinical_completeness",
                   "reasoning_quality", "safety_actionability", "visual_grounding"]
    
    # Overall averages
    all_scores = {d: [] for d in dimensions}
    # By severity
    by_severity = defaultdict(lambda: {d: [] for d in dimensions})
    # By question category
    by_category = defaultdict(lambda: {d: [] for d in dimensions})
    
    for r in judged_results:
        if r.get('skipped') or r.get('scores') is None:
            continue
        
        scores = r['scores']
        sev = r.get('severity', 'UNKNOWN')
        cat = r.get('question_category', 'unknown')
        
        for d in dimensions:
            val = scores.get(d)
            if val is not None:
                all_scores[d].append(val)
                by_severity[sev][d].append(val)
                by_category[cat][d].append(val)
    
    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    result = {
        'model': display_name,
        'n_judged': len([r for r in judged_results if not r.get('skipped')]),
        'overall': {d: round(mean(all_scores[d]), 3) for d in dimensions},
        'overall_average': round(mean([mean(all_scores[d]) for d in dimensions]), 3),
        'by_severity': {},
        'by_category': {}
    }
    
    for sev, scores_dict in by_severity.items():
        result['by_severity'][sev] = {d: round(mean(scores_dict[d]), 3) for d in dimensions}
    
    for cat, scores_dict in by_category.items():
        result['by_category'][cat] = {d: round(mean(scores_dict[d]), 3) for d in dimensions}
    
    return result


# ==============================================================================
# Reporting
# ==============================================================================
def generate_summary_report(all_aggregated: List[Dict], output_dir: str):
    """Generate summary tables and comparison reports."""
    dimensions = ["diagnostic_correctness", "clinical_completeness",
                   "reasoning_quality", "safety_actionability", "visual_grounding"]
    
    # Sort by overall average
    all_aggregated.sort(key=lambda x: x['overall_average'], reverse=True)
    
    # === CSV: Overall Leaderboard ===
    csv_file = os.path.join(output_dir, "llm_judge_leaderboard.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Model', 'Overall_Avg'] + [d.replace('_', ' ').title() for d in dimensions] + ['N_Judged']
        writer.writerow(header)
        for agg in all_aggregated:
            row = [agg['model'], agg['overall_average']]
            row += [agg['overall'].get(d, 0) for d in dimensions]
            row += [agg['n_judged']]
            writer.writerow(row)
    print(f"\nLeaderboard saved to: {csv_file}")
    
    # === CSV: By Severity ===
    sev_file = os.path.join(output_dir, "llm_judge_by_severity.csv")
    with open(sev_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Severity'] + [d.replace('_', ' ').title() for d in dimensions])
        for agg in all_aggregated:
            for sev in ['MILD', 'MODERATE', 'SEVERE']:
                if sev in agg.get('by_severity', {}):
                    row = [agg['model'], sev]
                    row += [agg['by_severity'][sev].get(d, 0) for d in dimensions]
                    writer.writerow(row)
    print(f"Severity breakdown saved to: {sev_file}")
    
    # === Print Summary ===
    print("\n" + "=" * 100)
    print("LLM-AS-JUDGE EVALUATION RESULTS")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 100)
    
    print(f"\n{'Model':<25} {'Overall':>8} {'Diag':>6} {'Clin':>6} {'Reas':>6} {'Safe':>6} {'VGrnd':>6} {'N':>5}")
    print("-" * 75)
    for agg in all_aggregated:
        o = agg['overall']
        print(f"{agg['model']:<25} {agg['overall_average']:>8.3f} "
              f"{o.get('diagnostic_correctness',0):>6.3f} "
              f"{o.get('clinical_completeness',0):>6.3f} "
              f"{o.get('reasoning_quality',0):>6.3f} "
              f"{o.get('safety_actionability',0):>6.3f} "
              f"{o.get('visual_grounding',0):>6.3f} "
              f"{agg['n_judged']:>5}")
    
    # === Save JSON summary ===
    summary_file = os.path.join(output_dir, "llm_judge_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'judge_model': JUDGE_MODEL,
            'n_sample_images': N_SAMPLE_IMAGES,
            'timestamp': datetime.now().isoformat(),
            'leaderboard': all_aggregated
        }, f, indent=2)
    print(f"\nFull summary saved to: {summary_file}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 80)
    print("PlantInquiryVQA - LLM-as-Judge Evaluation (Full Rebuttal Run)")
    print(f"Judge Model: {JUDGE_MODEL}")
    print(f"Sample Size: {N_SAMPLE_IMAGES} images")
    print(f"Models to evaluate: {len(MODELS_TO_EVALUATE)}")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: Pre-load ALL model results to find common images
    print("\n[1/5] Loading all model results...")
    loaded_models = {}  # display_name -> (model_name, results_list)
    all_image_sets = {}  # display_name -> set of image IDs
    
    for rel_path, display_name in MODELS_TO_EVALUATE.items():
        full_path = os.path.join(RESULTS_BASE, rel_path)
        
        if not os.path.exists(full_path):
            print(f"  SKIPPING {display_name} - file not found: {full_path}")
            continue
        
        try:
            print(f"  Loading {display_name} from {rel_path}...")
            model_name, results = load_model_results(full_path)
            loaded_models[display_name] = (model_name, results, rel_path)
            image_ids = {r['image_id'] for r in results}
            all_image_sets[display_name] = image_ids
            print(f"    -> {len(results)} images, model: {model_name}")
        except Exception as e:
            print(f"    ERROR loading {display_name}: {e}")
            continue
    
    if not loaded_models:
        print("ERROR: No models could be loaded!")
        return
    
    # Step 2: Find common images & select representative subset
    print(f"\n[2/5] Finding common images across {len(loaded_models)} models...")
    
    # Find images common to ALL loaded models
    common_images = None
    for display_name, img_set in all_image_sets.items():
        if common_images is None:
            common_images = img_set.copy()
        else:
            common_images &= img_set
    
    print(f"  Images common to ALL {len(loaded_models)} models: {len(common_images)}")
    
    # If common images are too few (test1a vs test2 have different images),
    # fall back to using per-model image sets
    use_per_model_sampling = False
    if len(common_images) < N_SAMPLE_IMAGES:
        print(f"  WARNING: Only {len(common_images)} common images (need {N_SAMPLE_IMAGES})")
        print(f"  Falling back to per-model image sampling...")
        use_per_model_sampling = True
        # Use the first model's images for sampling baseline
        first_display = list(loaded_models.keys())[0]
        _, first_results, _ = loaded_models[first_display]
        selected_images = select_representative_subset(first_results, N_SAMPLE_IMAGES)
    else:
        # Filter first model's results to only common images, then sample
        first_display = list(loaded_models.keys())[0]
        _, first_results, _ = loaded_models[first_display]
        common_results = [r for r in first_results if r['image_id'] in common_images]
        print(f"  Selecting {N_SAMPLE_IMAGES} representative images from {len(common_results)} common images...")
        selected_images = select_representative_subset(common_results, N_SAMPLE_IMAGES)
    
    # Save selected images for reproducibility
    selected_file = os.path.join(OUTPUT_DIR, "selected_images.json")
    with open(selected_file, 'w') as f:
        json.dump(selected_images, f, indent=2)
    print(f"  Selected images saved to: {selected_file}")
    
    # Step 3: Check which models already have results
    print(f"\n[3/5] Checking for existing results...")
    all_aggregated = []
    models_to_run = []
    
    for display_name, (model_name, results, rel_path) in loaded_models.items():
        model_dir = os.path.join(OUTPUT_DIR, display_name.replace(" ", "_").replace("/", "_"))
        final_file = os.path.join(model_dir, "judge_results.json")
        
        if os.path.exists(final_file):
            # Load existing results and re-aggregate with current selected images
            try:
                with open(final_file) as f:
                    existing = json.load(f)
                existing_results = existing.get('results', [])
                existing_images = {r['image_id'] for r in existing_results}
                selected_set = set(selected_images)
                
                # Check if existing results cover the selected images
                covered = selected_set & existing_images
                total_needed = sum(
                    len(r['conditions']) 
                    for r in results 
                    if r['image_id'] in selected_set
                )
                
                if len(covered) == len(selected_set):
                    print(f"  ✓ {display_name}: Already fully evaluated ({len(existing_results)} judgments)")
                    agg = aggregate_model_scores(display_name, existing_results)
                    all_aggregated.append(agg)
                    continue
                else:
                    print(f"  ~ {display_name}: Partially evaluated ({len(covered)}/{len(selected_set)} images)")
                    models_to_run.append(display_name)
            except Exception as e:
                print(f"  ! {display_name}: Error reading existing results: {e}")
                models_to_run.append(display_name)
        else:
            print(f"  ✗ {display_name}: No existing results, needs evaluation")
            models_to_run.append(display_name)
    
    # Step 4: Evaluate models that still need it
    if models_to_run:
        print(f"\n[4/5] Evaluating {len(models_to_run)} models (skipping {len(all_aggregated)} already done)...")
        
        for display_name in models_to_run:
            model_name, results, rel_path = loaded_models[display_name]
            
            # If using per-model sampling, select images specific to this model
            if use_per_model_sampling:
                model_image_ids = {r['image_id'] for r in results}
                model_selected = [img for img in selected_images if img in model_image_ids]
                if not model_selected:
                    # Fall back to sampling from this model's own images
                    print(f"\n--- Evaluating: {display_name} (per-model sampling) ---")
                    model_selected_list = select_representative_subset(results, N_SAMPLE_IMAGES)
                else:
                    model_selected_list = model_selected
            else:
                model_selected_list = selected_images
            
            print(f"\n--- Evaluating: {display_name} ---")
            print(f"  Source: {rel_path}")
            print(f"  {len(results)} total images, judging on {len(model_selected_list)} selected")
            
            try:
                agg = evaluate_model(model_name, display_name, results,
                                   model_selected_list, OUTPUT_DIR)
                all_aggregated.append(agg)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
    else:
        print(f"\n[4/5] All {len(all_aggregated)} models already evaluated! Regenerating reports...")
    
    # Step 5: Generate reports
    print(f"\n[5/5] Generating summary reports for {len(all_aggregated)} models...")
    generate_summary_report(all_aggregated, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print(f"Total models evaluated: {len(all_aggregated)}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
