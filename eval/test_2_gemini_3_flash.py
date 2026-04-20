import os
import json
import base64
import pandas as pd
from pathlib import Path
from typing import Dict, List
import time
from collections import defaultdict
from tqdm import tqdm
import traceback

from google import genai
from google.genai import types

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

DATASET_PATH = os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "dataset/plantinquiryvqa_test_subset.csv")
IMAGE_DIR = os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images")
OUTPUT_BASE_DIR = "test2_gemini_results"

SAVE_INTERVAL = 25

MODEL_CONFIG = {
    'model_name': 'gemini-3-flash-preview',
    'rpm_limit': 1000,
    'tpm_limit': 1000000,
    'rpd_limit': 10000,
    'output_dir': 'gemini3_flash'
}


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, rpm_limit, tpm_limit, rpd_limit):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        
        self.request_times = []
        self.token_usage = []
        self.daily_requests = []
        
        self.min_delay = 60.0 / rpm_limit if rpm_limit > 0 else 0.1
        
    def wait_if_needed(self, estimated_tokens=500):
        """Wait if rate limits would be exceeded."""
        current_time = time.time()
        
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
        self.daily_requests = [t for t in self.daily_requests if current_time - t < 86400]
        
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                time.sleep(wait_time + 0.1)
                current_time = time.time()
                self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        total_tokens = sum(tokens for _, tokens in self.token_usage)
        if total_tokens + estimated_tokens > self.tpm_limit:
            wait_time = 60 - (current_time - self.token_usage[0][0])
            if wait_time > 0:
                time.sleep(wait_time + 0.1)
                current_time = time.time()
                self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
        
        if len(self.daily_requests) >= self.rpd_limit:
            wait_time = 86400 - (current_time - self.daily_requests[0])
            if wait_time > 0:
                print(f"\nReached daily limit. Waiting {wait_time/3600:.1f} hours...")
                time.sleep(wait_time + 1)
                current_time = time.time()
                self.daily_requests = [t for t in self.daily_requests if current_time - t < 86400]
        
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


def load_dataset(n_images: int = 500) -> pd.DataFrame:
    """Load dataset."""
    df = pd.read_csv(DATASET_PATH)
    
    image_qa_counts = df.groupby('image_id').size().reset_index(name='qa_count')
    
    priority_images = image_qa_counts[image_qa_counts['qa_count'] >= 6]['image_id'].tolist()
    other_images = image_qa_counts[image_qa_counts['qa_count'] < 6]['image_id'].tolist()
    
    if len(priority_images) >= n_images:
        selected_images = priority_images[:n_images]
    else:
        selected_images = priority_images + other_images[:n_images - len(priority_images)]
    
    df_sample = df[df['image_id'].isin(selected_images)].copy()
    
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


def call_gemini_model(model_name: str, image_path: str, questions_so_far: List[str], 
                     rate_limiter: RateLimiter, gemini_client: genai.Client) -> str:
    """Call Gemini model with image and all questions so far (no answers)."""
    try:
        estimated_tokens = 500 + len(questions_so_far) * 50
        
        rate_limiter.wait_if_needed(estimated_tokens)
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        parts = []
        parts.append(types.Part.from_bytes(
            data=image_data,
            mime_type='image/jpeg'
        ))
        
        conversation_text = ""
        for i, q in enumerate(questions_so_far[:-1], 1):
            conversation_text += f"Q{i}: {q}\n"
        
        conversation_text += f"Q{len(questions_so_far)}: {questions_so_far[-1]}\nA{len(questions_so_far)}: "
        
        parts.append(conversation_text)
        
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=parts
        )
        
        rate_limiter.record_request(estimated_tokens)
        
        return response.text.strip()
        
    except Exception as e:
        rate_limiter.record_request(estimated_tokens)
        return f"ERROR: {str(e)}"


def run_test2_for_image(image_id: str, qa_pairs: List[Dict], 
                       image_path: str, rate_limiter: RateLimiter, 
                       gemini_client: genai.Client) -> Dict:
    """Run Test 2 for a single image - questions only, no answers."""
    
    model_name = MODEL_CONFIG['model_name']
    
    results = {
        'image_id': image_id,
        'crop': qa_pairs[0]['crop'],
        'disease': qa_pairs[0]['disease'],
        'severity': qa_pairs[0]['severity'],
        'num_questions': len(qa_pairs),
        'conditions': []
    }
    
    questions_so_far = []
    
    for i, qa in enumerate(qa_pairs, 1):
        questions_so_far.append(qa['question'])
        
        model_answer = call_gemini_model(model_name, image_path, questions_so_far, 
                                        rate_limiter, gemini_client)
        
        condition_result = {
            'question_number': i,
            'question': qa['question'],
            'question_category': qa['question_category'],
            'ground_truth': qa['answer'],
            'model_answer': model_answer,
            'questions_provided_so_far': questions_so_far.copy()
        }
        
        results['conditions'].append(condition_result)
    
    return results


def load_checkpoint(output_dir: str) -> tuple:
    """Load checkpoint if exists."""
    checkpoint_file = f"{output_dir}/checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return data.get('results', []), data.get('last_index', 0)
    return [], 0


def save_checkpoint(output_dir: str, results: List[Dict], last_index: int):
    """Save checkpoint."""
    checkpoint_data = {
        'results': results,
        'last_index': last_index,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f"{output_dir}/checkpoint.json", 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def main():
    """Main execution function."""
    print("="*80)
    print(f"TEST 2: QUESTION SCAFFOLDING - {MODEL_CONFIG['model_name']}")
    print("="*80)
    print("\nTest Description:")
    print("  - Provide image + all questions so far (Q1, Q2, Q3...)")
    print("  - No answers provided in context")
    print("  - Model must answer current question")
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, MODEL_CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    rate_limiter = RateLimiter(MODEL_CONFIG['rpm_limit'], MODEL_CONFIG['tpm_limit'], MODEL_CONFIG['rpd_limit'])
    
    print(f"\nRate limits: RPM={MODEL_CONFIG['rpm_limit']}, TPM={MODEL_CONFIG['tpm_limit']}, RPD={MODEL_CONFIG['rpd_limit']}")
    
    print("\nLoading dataset...")
    df = load_dataset(n_images=500)
    
    print("Organizing QA pairs by image...")
    image_data = organize_by_image(df)
    print(f"Found {len(image_data)} images")
    
    qa_counts = [len(qas) for qas in image_data.values()]
    print(f"QA distribution: min={min(qa_counts)}, max={max(qa_counts)}, avg={sum(qa_counts)/len(qa_counts):.1f}")
    
    all_results, start_index = load_checkpoint(output_dir)
    if start_index > 0:
        print(f"\nResuming from image {start_index + 1}")
    
    image_items = list(image_data.items())
    pbar = tqdm(image_items[start_index:], initial=start_index, total=len(image_data))
    
    for idx, (image_id, qa_pairs) in enumerate(pbar, start=start_index):
        pbar.set_description(f"Processing {idx+1}/{len(image_data)}")
        
        image_path = os.path.join(IMAGE_DIR, image_id)
        
        if not os.path.exists(image_path):
            print(f"\nWARNING: Image not found: {image_path}")
            continue
        
        try:
            result = run_test2_for_image(image_id, qa_pairs, image_path, 
                                        rate_limiter, gemini_client)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing {image_id}: {e}")
            traceback.print_exc()
            save_checkpoint(output_dir, all_results, idx)
            continue
        
        if (idx + 1) % SAVE_INTERVAL == 0:
            save_checkpoint(output_dir, all_results, idx + 1)
    
    print("\nSaving final results...")
    
    output_data = {
        'metadata': {
            'test_name': f'Test 2 - Question Scaffolding - {MODEL_CONFIG["model_name"]}',
            'test_description': 'Image + Questions only (no answers provided)',
            'model': MODEL_CONFIG['model_name'],
            'n_images': len(all_results),
            'total_qa_pairs': sum(len(r['conditions']) for r in all_results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': all_results
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCOMPLETED - Results saved to {output_dir}/results.json")
    print(f"Total images: {len(all_results)}")
    print(f"Total QA pairs: {sum(len(r['conditions']) for r in all_results)}")
    
    if os.path.exists(f"{output_dir}/checkpoint.json"):
        os.remove(f"{output_dir}/checkpoint.json")


if __name__ == "__main__":
    main()
