import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import gc
from pathlib import Path
import json
import os
import warnings
from collections import Counter, defaultdict
import re
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

BASE_PATH = os.environ.get("PLANTINQUIRY_HOME", ".")
TEST_CSV = f"{BASE_PATH}/Final_datasets/visual_cue_adaptive/vqa_test_500subset.csv"
IMAGES_DIR = f"{BASE_PATH}/images"
OUTPUT_DIR = f"{BASE_PATH}/Final_datasets/visual_cue_adaptive/test2_question_scaffolding"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*80)
print("TEST 2: QUESTION-ONLY PROTOCOL GUIDANCE - LLAVA-1.5-7B")
print("="*80)
print("\nHypothesis: Expert question structure alone (without answers) guides better diagnosis")
print("Testing: Image + Question List → Disease Prediction")
print("="*80)

class VQAEvaluator:
    def __init__(self):
        pass
    
    def tokenize(self, text):
        return str(text).lower().split()
    
    def normalize_answer(self, text):
        text = str(text).lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def exact_match(self, reference, hypothesis):
        return 1.0 if self.normalize_answer(reference) == self.normalize_answer(hypothesis) else 0.0
    
    def soft_accuracy(self, reference, hypothesis):
        ref_tokens = set(self.tokenize(reference))
        hyp_tokens = set(self.tokenize(hypothesis))
        
        if len(ref_tokens) == 0:
            return 0.0
        
        overlap = len(ref_tokens & hyp_tokens)
        return overlap / len(ref_tokens)
    
    def f1_score(self, reference, hypothesis):
        ref_tokens = self.tokenize(reference)
        hyp_tokens = self.tokenize(hypothesis)
        
        ref_counts = Counter(ref_tokens)
        hyp_counts = Counter(hyp_tokens)
        
        common = ref_counts & hyp_counts
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(hyp_tokens) if hyp_tokens else 0.0
        recall = num_same / len(ref_tokens) if ref_tokens else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1
    
    def disease_name_accuracy(self, disease_name, hypothesis):
        disease_norm = self.normalize_answer(disease_name)
        hyp_norm = self.normalize_answer(hypothesis)
        return 1.0 if disease_norm in hyp_norm else 0.0
    
    def evaluate(self, reference, hypothesis, disease_name):
        metrics = {
            'exact_match': self.exact_match(reference, hypothesis),
            'soft_accuracy': self.soft_accuracy(reference, hypothesis),
            'f1_score': self.f1_score(reference, hypothesis),
            'disease_name_accuracy': self.disease_name_accuracy(disease_name, hypothesis)
        }
        return metrics

print("\n[1/6] Loading test dataset...")
df_test = pd.read_csv(TEST_CSV)
print(f"✓ Loaded {len(df_test):,} test QA pairs")

print("\n[2/6] Setting image paths and filtering...")
df_test['image_path'] = df_test['image_name'].apply(lambda x: os.path.join(IMAGES_DIR, x))
df_test = df_test[df_test['image_path'].apply(os.path.exists)]
print(f"✓ Valid QA pairs: {len(df_test):,}")

print("\n[3/6] Grouping QA pairs by image and extracting questions...")
image_groups = defaultdict(list)

for _, row in df_test.iterrows():
    image_groups[row['image_name']].append({
        'question': row['question'],
        'answer': row['answer'],
        'crop': row['crop'],
        'disease': row['disease'],
        'severity': row['severity'],
        'image_path': row['image_path']
    })

print(f"✓ Grouped into {len(image_groups):,} unique images")
print(f"  Avg questions per image: {len(df_test)/len(image_groups):.2f}")

print("\n[4/6] Loading LLaVA-1.5-7B model...")

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

clean_memory()

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

print("✓ LLaVA-1.5-7B loaded successfully")

print("\n[5/6] Running TEST 2: Question-Only Scaffolding...")
print("  For each image:")
print("    - Extract all diagnostic questions (no answers)")
print("    - Present as numbered list")
print("    - Ask for disease diagnosis")
print("    - Compare accuracy vs TEST 1 baseline")

evaluator = VQAEvaluator()
results = []
checkpoint_file = f"{OUTPUT_DIR}/llava15_7b_test2_checkpoint.jsonl"

start_time = time.time()
total_inferences = 0

for image_name, qa_pairs in tqdm(image_groups.items(), desc="Processing images"):
    image_path = qa_pairs[0]['image_path']
    
    if not os.path.exists(image_path):
        continue
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"\n✗ Failed to load {image_name}: {e}")
        continue
    
    crop = qa_pairs[0]['crop']
    disease = qa_pairs[0]['disease']
    severity = qa_pairs[0]['severity']
    
    questions_list = [qa['question'] for qa in qa_pairs]
    
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_list)])
    
    prompt_text = f"""Consider these diagnostic questions about the plant in the image:

{questions_text}

Based on these diagnostic considerations, what disease is shown in the image? Provide the disease name."""
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    try:
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        inputs = processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(model.device, torch.float16) if torch.is_tensor(v) and v.dtype == torch.float else 
                     v.to(model.device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], outputs)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        predicted_disease = output_text[0].strip()
        
    except Exception as e:
        print(f"\n✗ Inference failed for {image_name}: {e}")
        predicted_disease = ""
    
    ground_truth = disease
    
    metrics = evaluator.evaluate(ground_truth, predicted_disease, disease)
    
    result = {
        'model': 'LLaVA-1.5-7B',
        'test': 'TEST2_Question_Scaffolding',
        'image_name': image_name,
        'crop': crop,
        'disease': disease,
        'severity': severity,
        'num_questions': len(questions_list),
        'questions_shown': questions_text,
        'predicted_disease': predicted_disease,
        'ground_truth_disease': ground_truth,
        **metrics
    }
    
    results.append(result)
    total_inferences += 1
    
    with open(checkpoint_file, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    if total_inferences % 50 == 0:
        elapsed = time.time() - start_time
        rate = total_inferences / elapsed
        print(f"\n  Processed {total_inferences:,} images | Rate: {rate:.2f} img/sec")

print(f"\n✓ Completed {total_inferences:,} inferences")

print("\n[6/6] Computing aggregates and saving results...")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("TEST 2 RESULTS - QUESTION-ONLY SCAFFOLDING")
print("="*80)

overall_metrics = {
    'exact_match': results_df['exact_match'].mean(),
    'soft_accuracy': results_df['soft_accuracy'].mean(),
    'f1_score': results_df['f1_score'].mean(),
    'disease_name_accuracy': results_df['disease_name_accuracy'].mean()
}

print(f"\nOverall Performance (Question-Only Scaffolding):")
print(f"  Exact Match:        {overall_metrics['exact_match']:.4f}")
print(f"  Soft Accuracy:      {overall_metrics['soft_accuracy']:.4f}")
print(f"  F1 Score:           {overall_metrics['f1_score']:.4f}")
print(f"  Disease Accuracy:   {overall_metrics['disease_name_accuracy']:.4f}")

by_severity = results_df.groupby('severity').agg({
    'exact_match': 'mean',
    'soft_accuracy': 'mean',
    'f1_score': 'mean',
    'disease_name_accuracy': 'mean'
}).reset_index()

print(f"\nPerformance by Severity:")
print(f"{'Severity':<12} {'Exact Match':<15} {'Soft Acc':<15} {'F1':<15} {'Disease Acc':<15}")
print("-"*80)
for _, row in by_severity.iterrows():
    print(f"{row['severity']:<12} {row['exact_match']:.4f}         {row['soft_accuracy']:.4f}         {row['f1_score']:.4f}         {row['disease_name_accuracy']:.4f}")

results_csv = f"{OUTPUT_DIR}/llava15_7b_test2_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\n✓ Detailed results: {results_csv}")

severity_summary = by_severity.to_dict('records')
summary = {
    'model': 'LLaVA-1.5-7B',
    'test': 'TEST2_Question_Scaffolding',
    'total_images': total_inferences,
    'overall_metrics': overall_metrics,
    'by_severity': severity_summary,
    'avg_questions_shown': results_df['num_questions'].mean(),
    'inference_time_minutes': (time.time() - start_time) / 60,
    'inference_rate': total_inferences / (time.time() - start_time)
}

with open(f"{OUTPUT_DIR}/llava15_7b_test2_summary.json", 'w') as f:
    json.dump(summary, indent=2, fp=f)

print("\n" + "="*80)
print("COMPARISON TO TEST 1 (Reference)")
print("="*80)
print("\nExpected pattern:")
print("  TEST 1 Q1 (Direct):        10-15% disease accuracy (no guidance)")
print("  TEST 2 (Questions-Only):   20-30% disease accuracy (protocol scaffolding)")
print("  TEST 1 Q7 (Full Context):  40-50% disease accuracy (with answers)")
print("\nScaffolding Benefit = TEST 2 - TEST 1 Q1")
print("Context Benefit     = TEST 1 Q7 - TEST 2")

print(f"\nTotal Time: {(time.time()-start_time)/60:.1f} minutes")
print(f"Inference Rate: {total_inferences/(time.time()-start_time):.2f} images/sec")

del model
del processor
clean_memory()

print("\n✓ TEST 2 COMPLETE: QUESTION-ONLY SCAFFOLDING - LLAVA-1.5-7B")
print("="*80)
