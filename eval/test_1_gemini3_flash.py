import os
import json
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import time
from collections import defaultdict
import re
from tqdm import tqdm
from datetime import datetime, timedelta

import openai
from google import genai
from google.genai import types

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

DATASET_PATH = os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "dataset/plantinquiryvqa_test_subset.csv")
IMAGE_DIR = os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images")
OUTPUT_DIR = "test1a_gemini3_results"

openai.api_key = OPENAI_API_KEY
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

USE_LLM_JUDGE = False
SAVE_INTERVAL = 25

# Rate limiting configuration for Gemini 3 Flash Preview
# Model: gemini-3-flash-preview
GEMINI_RPM_LIMIT = 1000
GEMINI_TPM_LIMIT = 1000000
GEMINI_RPD_LIMIT = 10000
GEMINI_MIN_DELAY = 60.0 / GEMINI_RPM_LIMIT  # ~0.06 seconds between requests


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, rpm_limit, tpm_limit, rpd_limit):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        
        self.request_times = []
        self.token_usage = []
        self.daily_requests = []
        
        self.min_delay = 60.0 / rpm_limit
        
    def wait_if_needed(self, estimated_tokens=500):
        """Wait if rate limits would be exceeded."""
        current_time = time.time()
        
        # Clean old entries (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
        
        # Clean daily entries (older than 24 hours)
        self.daily_requests = [t for t in self.daily_requests if current_time - t < 86400]
        
        # Check RPM limit
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time + 0.1)
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check TPM limit
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.tpm_limit:
            wait_time = 60 - (current_time - self.token_usage[0][0])
            if wait_time > 0:
                time.sleep(wait_time + 0.1)
                current_time = time.time()
                self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
        
        # Check RPD limit
        if len(self.daily_requests) >= self.rpd_limit:
            wait_time = 86400 - (current_time - self.daily_requests[0])
            if wait_time > 0:
                print(f"\nReached daily limit. Waiting {wait_time/3600:.1f} hours...")
                time.sleep(wait_time + 1)
                current_time = time.time()
                self.daily_requests = [t for t in self.daily_requests if current_time - t < 86400]
        
        # Minimum delay between requests
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < self.min_delay:
                time.sleep(self.min_delay - time_since_last)
        
    def record_request(self, tokens_used=500):
        """Record a completed request."""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_usage.append((current_time, tokens_used))
        self.daily_requests.append(current_time)


gemini_limiter = RateLimiter(GEMINI_RPM_LIMIT, GEMINI_TPM_LIMIT, GEMINI_RPD_LIMIT)


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_dataset(n_images: int = 500) -> pd.DataFrame:
    """Load dataset, prioritizing diseased images with full QA sequences."""
    df = pd.read_csv(DATASET_PATH)
    
    image_qa_counts = df.groupby('image_id').size().reset_index(name='qa_count')
    
    priority_images = image_qa_counts[image_qa_counts['qa_count'] >= 6]['image_id'].tolist()
    other_images = image_qa_counts[image_qa_counts['qa_count'] < 6]['image_id'].tolist()
    
    if len(priority_images) >= n_images:
        selected_images = priority_images[:n_images]
    else:
        selected_images = priority_images + other_images[:n_images - len(priority_images)]
    
    df_sample = df[df['image_id'].isin(selected_images)].copy()
    
    print(f"Loaded {len(selected_images)} images with {len(df_sample)} QA pairs")
    print(f"  Priority (6+ questions): {len([img for img in selected_images if img in priority_images])}")
    print(f"  Other (<6 questions): {len([img for img in selected_images if img not in priority_images])}")
    
    return df_sample


def organize_by_image(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Organize QA pairs by image."""
    image_data = defaultdict(list)
    
    for _, row in df.iterrows():
        image_id = row['image_id']
        qa_pair = {
            'question_number': row['question_number'],
            'question': row['question'],
            'answer': row['answer'],
            'question_category': row['question_category'],
            'crop': row['crop'],
            'disease': row['disease'],
            'severity': row['severity']
        }
        image_data[image_id].append(qa_pair)
    
    for image_id in image_data:
        image_data[image_id] = sorted(image_data[image_id], key=lambda x: x['question_number'])
    
    return dict(image_data)


def call_gpt4o(image_base64: str, conversation_history: List[Dict], current_question: str) -> str:
    """Call GPT-4o API with image and conversation context."""
    try:
        messages = []
        
        first_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]
        
        if conversation_history:
            first_content.append({
                "type": "text",
                "text": conversation_history[0]['question']
            })
            messages.append({
                "role": "user",
                "content": first_content
            })
            messages.append({
                "role": "assistant",
                "content": conversation_history[0]['answer']
            })
            
            for turn in conversation_history[1:]:
                messages.append({
                    "role": "user",
                    "content": turn['question']
                })
                messages.append({
                    "role": "assistant",
                    "content": turn['answer']
                })
        else:
            first_content.append({
                "type": "text",
                "text": current_question
            })
            messages.append({
                "role": "user",
                "content": first_content
            })
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        
        messages.append({
            "role": "user",
            "content": current_question
        })
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"GPT-4o API Error: {e}")
        return f"ERROR: {str(e)}"


def call_gemini3(image_path: str, conversation_history: List[Dict], current_question: str) -> str:
    """Call Gemini 3 Flash Preview API with rate limiting."""
    try:
        # Estimate tokens (rough approximation)
        estimated_tokens = 500
        if conversation_history:
            estimated_tokens += sum(len(turn['question'].split()) + len(turn['answer'].split()) 
                                   for turn in conversation_history) * 2
        
        gemini_limiter.wait_if_needed(estimated_tokens)
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        parts = []
        
        parts.append(types.Part.from_bytes(
            data=image_data,
            mime_type='image/jpeg'
        ))
        
        conversation_text = ""
        if conversation_history:
            for turn in conversation_history:
                conversation_text += f"Q: {turn['question']}\nA: {turn['answer']}\n\n"
        
        conversation_text += f"Q: {current_question}\nA: "
        
        parts.append(conversation_text)
        
        response = gemini_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=parts
        )
        
        gemini_limiter.record_request(estimated_tokens)
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Gemini 3 Flash Preview API Error: {e}")
        gemini_limiter.record_request(estimated_tokens)
        return f"ERROR: {str(e)}"


def extract_disease_name(text: str) -> List[str]:
    """Extract disease names from text."""
    text_lower = text.lower()
    
    disease_keywords = [
        'blight', 'spot', 'rot', 'mildew', 'rust', 'wilt', 'mosaic',
        'virus', 'scab', 'canker', 'mold', 'mould', 'bacterial', 'fungal',
        'healthy', 'necrosis', 'streak', 'curl', 'leaf curl', 'powdery',
        'downy', 'anthracnose', 'septoria', 'cercospora', 'alternaria'
    ]
    
    found_diseases = []
    for keyword in disease_keywords:
        if keyword in text_lower:
            found_diseases.append(keyword)
    
    return found_diseases


def compute_visual_grounding(answer: str) -> Tuple[bool, int]:
    """Check if answer contains visual descriptors."""
    visual_cues = [
        'spot', 'lesion', 'discoloration', 'wilting', 'yellowing', 'browning',
        'necrotic', 'chlorotic', 'margin', 'leaf', 'stem', 'tissue',
        'color', 'pattern', 'visible', 'appearance', 'surface', 'texture',
        'uniform', 'irregular', 'circular', 'elongated', 'scattered',
        'green', 'brown', 'yellow', 'white', 'black', 'gray', 'grey',
        'dried', 'withered', 'deformed', 'curled', 'twisted'
    ]
    
    answer_lower = answer.lower()
    count = sum(1 for cue in visual_cues if cue in answer_lower)
    
    return count > 0, count


def compute_metrics(prediction: str, reference: str, gt_disease: str) -> Dict:
    """Compute all metrics for a single prediction."""
    metrics = {}
    
    metrics['exact_match'] = int(prediction.strip().lower() == reference.strip().lower())
    
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if len(ref_tokens) > 0:
        overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
        metrics['soft_accuracy'] = int(overlap > 0.5)
    else:
        metrics['soft_accuracy'] = 0
    
    if len(pred_tokens) > 0 and len(ref_tokens) > 0:
        precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
        recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
        if precision + recall > 0:
            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        else:
            metrics['f1_score'] = 0.0
    else:
        metrics['f1_score'] = 0.0
    
    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()
    smoothie = SmoothingFunction().method4
    metrics['bleu_1'] = sentence_bleu(reference_tokens, prediction_tokens, 
                                       weights=(1, 0, 0, 0), smoothing_function=smoothie)
    metrics['bleu_4'] = sentence_bleu(reference_tokens, prediction_tokens, 
                                       weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    try:
        rouge = Rouge()
        scores = rouge.get_scores(prediction, reference)[0]
        metrics['rouge_l'] = scores['rouge-l']['f']
    except:
        metrics['rouge_l'] = 0.0
    
    try:
        metrics['meteor'] = meteor_score([reference.split()], prediction.split())
    except:
        metrics['meteor'] = 0.0
    
    metrics['cider'] = metrics['bleu_4']
    
    pred_diseases = extract_disease_name(prediction)
    gt_disease_clean = str(gt_disease).lower() if pd.notna(gt_disease) else ''
    metrics['disease_mentioned'] = int(any(disease in gt_disease_clean for disease in pred_diseases))
    
    is_grounded, vg_count = compute_visual_grounding(prediction)
    metrics['visual_grounding'] = int(is_grounded)
    metrics['visual_cue_count'] = vg_count
    
    metrics['answer_length'] = len(prediction.split())
    
    return metrics


def run_test1a_for_image(image_id: str, qa_pairs: List[Dict], image_path: str, 
                         image_base64: str) -> Dict:
    """Run Test 1A for a single image - Gemini 3 Flash Preview only."""
    
    results = {
        'image_id': image_id,
        'crop': qa_pairs[0]['crop'],
        'disease': qa_pairs[0]['disease'],
        'severity': qa_pairs[0]['severity'],
        'num_questions': len(qa_pairs),
        'conditions': []
    }
    
    for i, qa in enumerate(qa_pairs, 1):
        conversation_history = []
        for j in range(i-1):
            conversation_history.append({
                'question': qa_pairs[j]['question'],
                'answer': qa_pairs[j]['answer']
            })
        
        gemini3_answer = call_gemini3(image_path, conversation_history, qa['question'])
        
        gemini3_metrics = compute_metrics(gemini3_answer, qa['answer'], qa['disease'])
        
        condition_result = {
            'question_number': i,
            'question': qa['question'],
            'question_category': qa['question_category'],
            'ground_truth': qa['answer'],
            'gemini3': {
                'answer': gemini3_answer,
                'metrics': gemini3_metrics
            }
        }
        
        results['conditions'].append(condition_result)
    
    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """Aggregate metrics across all images and conditions."""
    
    aggregated = {
        'per_position': defaultdict(lambda: defaultdict(list)),
        'per_severity': defaultdict(lambda: defaultdict(list)),
        'summary': {}
    }
    
    for image_result in all_results:
        severity = str(image_result.get('severity', 'unknown'))
        
        for condition in image_result['conditions']:
            pos = condition['question_number']
            
            for metric_name, value in condition['gemini3']['metrics'].items():
                aggregated['per_position'][f'Q{pos}']['gemini3_' + metric_name].append(value)
                aggregated['per_severity'][severity]['gemini3_' + metric_name].append(value)
    
    summary = {
        'by_position': {},
        'by_severity': {},
        'overall': {},
        'cumulative_improvement': {}
    }
    
    for pos, metrics_dict in aggregated['per_position'].items():
        summary['by_position'][pos] = {}
        for metric_name, values in metrics_dict.items():
            summary['by_position'][pos][metric_name] = sum(values) / len(values) if values else 0
    
    for severity, metrics_dict in aggregated['per_severity'].items():
        summary['by_severity'][severity] = {}
        for metric_name, values in metrics_dict.items():
            summary['by_severity'][severity][metric_name] = sum(values) / len(values) if values else 0
    
    all_metrics = defaultdict(list)
    for pos_data in aggregated['per_position'].values():
        for metric_name, values in pos_data.items():
            all_metrics[metric_name].extend(values)
    
    for metric_name, values in all_metrics.items():
        summary['overall'][metric_name] = sum(values) / len(values) if values else 0
    
    positions = sorted([int(p[1:]) for p in summary['by_position'].keys()])
    if len(positions) >= 2:
        q1_key = f'Q{positions[0]}'
        qn_key = f'Q{positions[-1]}'
        
        for metric in ['soft_accuracy', 'f1_score', 'visual_grounding', 'bleu_4']:
            metric_name = f'gemini3_{metric}'
            q1_val = summary['by_position'][q1_key].get(metric_name, 0)
            qn_val = summary['by_position'][qn_key].get(metric_name, 0)
            if q1_val > 0:
                improvement = ((qn_val - q1_val) / q1_val) * 100
                summary['cumulative_improvement'][metric] = improvement
    
    return summary


def load_checkpoint() -> Tuple[List[Dict], int]:
    """Load checkpoint if exists."""
    checkpoint_file = f"{OUTPUT_DIR}/checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return data.get('results', []), data.get('last_index', 0)
    return [], 0


def save_checkpoint(results: List[Dict], last_index: int):
    """Save checkpoint."""
    checkpoint_data = {
        'results': results,
        'last_index': last_index,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f"{OUTPUT_DIR}/checkpoint.json", 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def main():
    """Main execution function."""
    print("="*80)
    print("TEST 1A: GEMINI 3 FLASH PREVIEW EVALUATION (500 IMAGES)")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[1/4] Loading dataset...")
    df = load_dataset(n_images=500)
    
    print("\n[2/4] Organizing QA pairs by image...")
    image_data = organize_by_image(df)
    print(f"Found {len(image_data)} images")
    
    qa_counts = [len(qas) for qas in image_data.values()]
    print(f"QA distribution: min={min(qa_counts)}, max={max(qa_counts)}, avg={sum(qa_counts)/len(qa_counts):.1f}")
    
    all_results, start_index = load_checkpoint()
    if start_index > 0:
        print(f"\nResuming from image {start_index + 1}")
    
    print(f"\n[3/4] Running Test 1A evaluations...")
    print(f"Rate limits: RPM={GEMINI_RPM_LIMIT}, TPM={GEMINI_TPM_LIMIT}, RPD={GEMINI_RPD_LIMIT}")
    
    image_items = list(image_data.items())
    pbar = tqdm(image_items[start_index:], initial=start_index, total=len(image_data), desc="Processing images")
    
    for idx, (image_id, qa_pairs) in enumerate(pbar, start=start_index):
        pbar.set_description(f"Image {idx+1}/{len(image_data)} - {image_id[:12]}")
        
        image_path = os.path.join(IMAGE_DIR, image_id)
        
        if not os.path.exists(image_path):
            print(f"\nWARNING: Image not found: {image_path}")
            continue
        
        image_base64 = encode_image(image_path)
        
        try:
            result = run_test1a_for_image(image_id, qa_pairs, image_path, image_base64)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {image_id}: {e}")
            save_checkpoint(all_results, idx)
            continue
        
        if (idx + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(all_results, idx + 1)
            pbar.set_postfix({"saved": f"{idx + 1} images"})
    
    print("\n[4/4] Aggregating results...")
    summary = aggregate_results(all_results)
    
    output_data = {
        'metadata': {
            'test_name': 'Test 1A - Gemini 3 Flash Preview Evaluation',
            'n_images': len(all_results),
            'total_qa_pairs': sum(len(r['conditions']) for r in all_results),
            'model': 'gemini-3-flash-preview',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detailed_results': all_results,
        'summary': summary
    }
    
    with open(f"{OUTPUT_DIR}/test1a_gemini3_results.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY RESULTS - GEMINI 3 FLASH PREVIEW")
    print("="*80)
    
    print(f"\nTotal Images Processed: {len(all_results)}")
    print(f"Total QA Pairs Evaluated: {sum(len(r['conditions']) for r in all_results)}")
    
    print("\n--- Overall Performance ---")
    overall_data = summary['overall']
    print(f"Soft Accuracy: {overall_data.get('gemini3_soft_accuracy', 0):.3f}")
    print(f"F1 Score: {overall_data.get('gemini3_f1_score', 0):.3f}")
    print(f"Visual Grounding: {overall_data.get('gemini3_visual_grounding', 0):.3f}")
    print(f"BLEU-4: {overall_data.get('gemini3_bleu_4', 0):.3f}")
    print(f"ROUGE-L: {overall_data.get('gemini3_rouge_l', 0):.3f}")
    
    print("\n--- Cumulative Improvement (Q1 → Qn) ---")
    for metric, value in summary['cumulative_improvement'].items():
        print(f"{metric}: {value:+.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}/test1a_gemini3_results.json")
    print(f"{'='*80}")
    
    if os.path.exists(f"{OUTPUT_DIR}/checkpoint.json"):
        os.remove(f"{OUTPUT_DIR}/checkpoint.json")


if __name__ == "__main__":
    main()