"""
Cascading Context Test (Autoregressive History)
================================================
Addresses Reviewer W3: "guided context test uses ground-truth history"

This test feeds the MODEL'S OWN previous answers as conversation history,
NOT ground-truth. This simulates real deployment where early errors
can cascade into later turns.

Comparison:
  - Guided (Test 1a):    GT history      → Upper bound
  - Cascading (THIS):    Model history   → Realistic scenario
  - Scaffolded (Test 2): No history      → Lower bound
  - Unconstrained:       No CoI at all   → Worst case

Models: Top 5 from Table 2 (excluding Qwen)
Images: 50 (from test subset)
"""

import os
import json
import base64
import pandas as pd
import time
import random
import requests
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from google import genai
from google.genai import types

# ==============================================================================
# CONFIG
# ==============================================================================

BASE_DIR = os.environ.get("PLANTINQUIRY_HOME", ".")
DATASET_CSV = os.path.join(BASE_DIR, "dataset/plantinquiryvqa_test_subset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "eval/cascading_context_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Multiple image directories to search
IMAGE_DIRS = [
    os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images"),
    os.path.join(BASE_DIR, "images"),
]

N_IMAGES = 50

# API Keys
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Top 5 models from Table 2 (by Disease Accuracy, excluding Qwen)
MODELS = {
    "gemini3_flash": {
        "display_name": "Gemini-3-Flash",
        "type": "gemini",
        "model_id": "gemini-3-flash-preview",
    },
    "gemini25_pro": {
        "display_name": "Gemini-2.5-Pro",
        "type": "gemini",
        "model_id": "gemini-2.5-pro",
    },
    "seed_1_6_flash": {
        "display_name": "Seed-1.6-Flash",
        "type": "openrouter",
        "model_id": "bytedance-seed/seed-1.6-flash",
    },
    "llama_3_2_90b": {
        "display_name": "Llama-3.2-90B-Vision",
        "type": "openrouter",
        "model_id": "meta-llama/llama-3.2-90b-vision-instruct",
    },
    "llama_4_maverick": {
        "display_name": "Llama-4-Maverick",
        "type": "openrouter",
        "model_id": "meta-llama/llama-4-maverick",
    },
}


# ==============================================================================
# IMAGE FINDER
# ==============================================================================

def find_image(image_id):
    """Search for image across multiple directories."""
    for d in IMAGE_DIRS:
        path = os.path.join(d, image_id)
        if os.path.exists(path):
            return path
    return None


# ==============================================================================
# RETRY DECORATOR
# ==============================================================================

def retry_with_backoff(func, *args, retries=3, base_delay=5, **kwargs):
    """Retry a function with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if attempt == retries:
                print(f"  FAILED after {retries+1} attempts: {err_str[:120]}")
                return f"ERROR: {err_str}"
            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"  Retry {attempt+1}/{retries} in {delay:.1f}s: {err_str[:100]}...")
            time.sleep(delay)


# ==============================================================================
# MODEL CALLING FUNCTIONS
# ==============================================================================

def call_gemini(image_path: str, conversation_history: List[Dict],
                current_question: str, model_id: str) -> str:
    """Call Gemini model with image and cascading conversation history."""
    with open(image_path, 'rb') as f:
        image_data = f.read()

    parts = []
    parts.append(types.Part.from_bytes(data=image_data, mime_type='image/jpeg'))

    # Build conversation text with MODEL's own previous answers
    conversation_text = ""
    if conversation_history:
        for turn in conversation_history:
            conversation_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"

    conversation_text += f"Q: {current_question}\nA: "
    parts.append(conversation_text)

    response = gemini_client.models.generate_content(
        model=model_id,
        contents=parts
    )
    return response.text.strip()


def encode_image_to_url(image_path: str) -> str:
    """Encode image to base64 data URL for OpenRouter."""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def call_openrouter(model_id: str, image_url: str,
                    conversation_history: List[Dict],
                    current_question: str) -> str:
    """Call OpenRouter model with cascading conversation history."""
    messages = []

    if conversation_history:
        # First message includes image + first question
        first_message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": conversation_history[0]['question']}
            ]
        }
        messages.append(first_message)
        messages.append({
            "role": "assistant",
            "content": conversation_history[0]['answer']
        })

        # Subsequent turns
        for turn in conversation_history[1:]:
            messages.append({"role": "user", "content": turn['question']})
            messages.append({"role": "assistant", "content": turn['answer']})

        # Current question
        messages.append({"role": "user", "content": current_question})
    else:
        # First question — include image
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": current_question}
            ]
        })

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={"model": model_id, "messages": messages},
        timeout=120
    )

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"HTTP {response.status_code}: {response.text[:300]}")


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_prepare_data():
    """Load dataset and select 50 images with valid image files."""
    df = pd.read_csv(DATASET_CSV)

    # Organize by image
    image_data = defaultdict(list)
    for _, row in df.iterrows():
        if pd.isna(row.get('question_number')) or pd.isna(row.get('question')):
            continue
        image_id = row['image_id']
        qa_pair = {
            'question_number': int(row['question_number']),
            'question': str(row['question']),
            'answer': str(row.get('answer', '')),
            'question_category': str(row.get('question_category', '')),
            'crop': str(row.get('crop', '')),
            'disease': str(row.get('disease', '')),
            'severity': str(row.get('severity', '')),
        }
        image_data[image_id].append(qa_pair)

    # Sort questions by number within each image
    for image_id in image_data:
        image_data[image_id] = sorted(image_data[image_id],
                                       key=lambda x: x['question_number'])

    # Filter for images that exist on disk (check multiple directories)
    valid_images = {}
    for image_id, qa_pairs in image_data.items():
        img_path = find_image(image_id)
        if img_path is not None:
            valid_images[image_id] = qa_pairs

    print(f"Found {len(valid_images)} valid images out of {len(image_data)} in CSV")

    # Select N images: prioritize diseased images with 6+ questions
    priority = {k: v for k, v in valid_images.items()
                if len(v) >= 6 and v[0].get('disease', 'healthy').lower() != 'healthy'}
    healthy = {k: v for k, v in valid_images.items()
               if v[0].get('disease', 'healthy').lower() == 'healthy'}
    other = {k: v for k, v in valid_images.items()
             if k not in priority and k not in healthy}

    selected = {}
    # First add diseased images with full chains
    for k, v in list(priority.items()):
        if len(selected) >= N_IMAGES:
            break
        selected[k] = v
    # Then other images
    for k, v in list(other.items()):
        if len(selected) >= N_IMAGES:
            break
        selected[k] = v
    # Then healthy if still needed
    for k, v in list(healthy.items()):
        if len(selected) >= N_IMAGES:
            break
        selected[k] = v

    total_qs = sum(len(v) for v in selected.values())
    print(f"Selected {len(selected)} images ({total_qs} total QAs) for cascading context test")
    return selected


# ==============================================================================
# CHECKPOINTING
# ==============================================================================

def load_checkpoint(model_key):
    """Load checkpoint for a model if it exists."""
    ckpt_path = os.path.join(OUTPUT_DIR, f"{model_key}_checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'r') as f:
            data = json.load(f)
        print(f"  Loaded checkpoint: {len(data['results'])} images done")
        return data['results'], set(data['completed_images'])
    return [], set()


def save_checkpoint(model_key, results, completed_images):
    """Save checkpoint."""
    ckpt_path = os.path.join(OUTPUT_DIR, f"{model_key}_checkpoint.json")
    with open(ckpt_path, 'w') as f:
        json.dump({
            'results': results,
            'completed_images': list(completed_images)
        }, f, indent=2, default=str)


# ==============================================================================
# CASCADING CONTEXT EVALUATION
# ==============================================================================

def run_cascading_for_image(model_key, model_config, image_id, qa_pairs, image_path):
    """
    Run cascading context test for one image.

    KEY DIFFERENCE from Guided (Test 1a):
    - Test 1a: conversation_history[j]['answer'] = qa_pairs[j]['answer']  (GT!)
    - THIS:    conversation_history[j]['answer'] = model_answer           (model's own!)
    """
    image_result = {
        'image_id': image_id,
        'crop': qa_pairs[0]['crop'],
        'disease': qa_pairs[0]['disease'],
        'severity': qa_pairs[0]['severity'],
        'num_questions': len(qa_pairs),
        'qa_results': []
    }

    # This accumulates the MODEL'S OWN answers as history
    model_conversation_history = []

    # Pre-encode image for OpenRouter (do once per image)
    image_url = None
    if model_config['type'] == 'openrouter':
        image_url = encode_image_to_url(image_path)

    for i, qa in enumerate(qa_pairs):
        # Call the model with its OWN accumulated history
        if model_config['type'] == 'gemini':
            model_answer = retry_with_backoff(
                call_gemini, image_path,
                model_conversation_history, qa['question'],
                model_config['model_id']
            )
        else:  # openrouter
            model_answer = retry_with_backoff(
                call_openrouter, model_config['model_id'], image_url,
                model_conversation_history, qa['question']
            )

        if model_answer is None:
            model_answer = "ERROR: No response"

        # Record result
        qa_result = {
            'question_number': qa['question_number'],
            'question_category': qa['question_category'],
            'question': qa['question'],
            'ground_truth': qa['answer'],
            'model_answer': model_answer,
            'history_length': len(model_conversation_history),
        }
        image_result['qa_results'].append(qa_result)

        # *** KEY: Add MODEL's OWN answer to history (NOT ground truth!) ***
        model_conversation_history.append({
            'question': qa['question'],
            'answer': model_answer  # <-- MODEL'S OWN ANSWER
        })

        time.sleep(0.8)  # Rate limiting

    return image_result


def run_model(model_key, model_config, image_data):
    """Run cascading test for one model across all images."""
    display_name = model_config['display_name']
    print(f"\n{'='*60}")
    print(f"  MODEL: {display_name} (Cascading Context)")
    print(f"{'='*60}")

    results, completed = load_checkpoint(model_key)
    remaining = {k: v for k, v in image_data.items() if k not in completed}

    print(f"  Remaining: {len(remaining)} images")

    pbar = tqdm(remaining.items(), desc=display_name,
                total=len(remaining), unit="img")

    for idx, (image_id, qa_pairs) in enumerate(pbar):
        image_path = find_image(image_id)
        if not image_path:
            print(f"  SKIP {image_id}: not found")
            continue

        pbar.set_postfix_str(
            f"{image_id[:12]}... ({qa_pairs[0]['crop']}/{qa_pairs[0]['disease']})")

        try:
            result = run_cascading_for_image(
                model_key, model_config, image_id, qa_pairs, image_path
            )
            results.append(result)
            completed.add(image_id)
        except Exception as e:
            print(f"  ERROR on {image_id}: {e}")
            continue

        # Save checkpoint every 10 images
        if (idx + 1) % 10 == 0:
            save_checkpoint(model_key, results, completed)

    # Final save
    save_checkpoint(model_key, results, completed)
    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   CASCADING CONTEXT TEST — Model-Generated History                 ║")
    print("║   Addresses W3: 'guided context uses ground-truth history'         ║")
    print("║   Models use their OWN previous answers as conversation history    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    print(f"Models: {[m['display_name'] for m in MODELS.values()]}")
    print(f"Images: {N_IMAGES}\n")

    # Load data
    image_data = load_and_prepare_data()

    all_results = {}

    for model_key, model_config in MODELS.items():
        results = run_model(model_key, model_config, image_data)
        all_results[model_key] = {
            "display_name": model_config['display_name'],
            "n_images": len(results),
            "n_qa_total": sum(len(r['qa_results']) for r in results),
            "results": results
        }

    # Save final combined results
    output_path = os.path.join(OUTPUT_DIR, "cascading_context_all_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nAll results saved to: {output_path}")

    # Quick summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_key, data in all_results.items():
        n_err = 0
        n_total = 0
        for img_result in data['results']:
            for qa in img_result['qa_results']:
                n_total += 1
                if str(qa.get('model_answer', '')).startswith('ERROR'):
                    n_err += 1
        print(f"  {data['display_name']}: {data['n_images']} images, "
              f"{data['n_qa_total']} QAs, {n_err} errors")

    print(f"\n✅ Phase 1 complete. Run LLM-as-Judge evaluation next.")
    print(f"   Results at: {output_path}")


if __name__ == "__main__":
    main()
