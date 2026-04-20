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
import requests

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

DATASET_PATH = os.path.join(os.environ.get("PLANTINQUIRY_HOME", "."), "dataset/plantinquiryvqa_test_subset.csv")
IMAGE_DIR = os.environ.get("PLANTINQUIRY_IMAGE_DIR", "images")
OUTPUT_BASE_DIR = "test2_free_models_results"

SAVE_INTERVAL = 25

MODEL_CONFIG = {
    'model_name': 'qwen/qwen-2.5-vl-7b-instruct:free',
    'output_dir': 'qwen25_vl_7b_free'
}


def encode_image_to_url(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"


def load_dataset(n_images: int = 500) -> pd.DataFrame:
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


def call_qwen_model(model_name: str, image_url: str, questions_so_far: List[str]) -> str:
    try:
        conversation_text = ""
        for i, q in enumerate(questions_so_far[:-1], 1):
            conversation_text += f"Q{i}: {q}\n"
        
        conversation_text += f"Q{len(questions_so_far)}: {questions_so_far[-1]}\nA{len(questions_so_far)}: "
        
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                },
                {
                    "type": "text",
                    "text": conversation_text
                }
            ]
        }]
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "messages": messages
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"ERROR: HTTP {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"ERROR: {str(e)}"


def run_test2_for_image(image_id: str, qa_pairs: List[Dict], image_path: str) -> Dict:
    model_name = MODEL_CONFIG['model_name']
    image_url = encode_image_to_url(image_path)
    
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
        
        model_answer = call_qwen_model(model_name, image_url, questions_so_far)
        
        time.sleep(1)
        
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
    checkpoint_file = f"{output_dir}/checkpoint.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return data.get('results', []), data.get('last_index', 0)
    return [], 0


def save_checkpoint(output_dir: str, results: List[Dict], last_index: int):
    checkpoint_data = {
        'results': results,
        'last_index': last_index,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f"{output_dir}/checkpoint.json", 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


def main():
    print("="*80)
    print(f"TEST 2: QUESTION SCAFFOLDING - {MODEL_CONFIG['model_name']}")
    print("="*80)
    print("\nTest Description:")
    print("  - Provide image + all questions so far (Q1, Q2, Q3...)")
    print("  - No answers provided in context")
    print("  - Model must answer current question")
    print("\nNOTE: This is a FREE model - no API costs!")
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, MODEL_CONFIG['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
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
            result = run_test2_for_image(image_id, qa_pairs, image_path)
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
            'cost': 'FREE',
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
    print(f"Total cost: $0.00 (FREE model!)")
    
    if os.path.exists(f"{output_dir}/checkpoint.json"):
        os.remove(f"{output_dir}/checkpoint.json")


if __name__ == "__main__":
    main()
