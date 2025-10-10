#!/usr/bin/env python3
"""
Script to evaluate our GPT-2 victim model on the adversarial dataset.
"""

import json
import argparse
import sys
import os
from typing import Dict, List, Any
from tqdm import tqdm

# Add the parent directory to path to import our victim model
sys.path.append('/data/arpit/code/')
from gpt2_victim_model import GPT2QAVictimModel

def load_adversarial_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """Load the adversarial dataset."""
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    examples = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                examples.append({
                    'id': qa['id'],
                    'question': qa['question'],
                    'context': paragraph['context'],
                    'adversarial_context': qa.get('adversarial_context', paragraph['context']),
                    'original_answer': qa['answers'][0]['text'] if qa['answers'] else '',
                    'adversarial_answers': qa.get('adversarial_answers', []),
                    'is_successful_attack': qa.get('is_successful_attack', False),
                    'top_tokens': qa.get('top_tokens', [])
                })
    
    return examples

def evaluate_gpt2_victim_on_examples(victim_model: GPT2QAVictimModel, examples: List[Dict]) -> Dict[str, Any]:
    """Evaluate our GPT-2 victim model on the adversarial examples."""
    results = {
        'total_examples': len(examples),
        'correct_on_original': 0,
        'correct_on_adversarial': 0,
        'robust_examples': 0,  # Correct on both original and adversarial
        'vulnerable_examples': 0,  # Correct on original but wrong on adversarial
        'failed_examples': 0,  # Wrong on both
        'improved_examples': 0,  # Wrong on original but correct on adversarial
        'detailed_results': []
    }
    
    for example in tqdm(examples, desc="Evaluating GPT-2 victim model"):
        try:
            # Get predictions on original context
            original_pred = victim_model.get_answer(example['question'], example['context'], bit_width=8)
            
            # Get predictions on adversarial context
            adversarial_pred = victim_model.get_answer(example['question'], example['adversarial_context'], bit_width=8)
            
            # Check if predictions are correct (simple string matching for now)
            original_correct = is_answer_correct(original_pred, example['original_answer'])
            adversarial_correct = is_answer_correct(adversarial_pred, example['original_answer'])
            
            # Update counters
            if original_correct:
                results['correct_on_original'] += 1
            if adversarial_correct:
                results['correct_on_adversarial'] += 1
            
            if original_correct and adversarial_correct:
                results['robust_examples'] += 1
            elif original_correct and not adversarial_correct:
                results['vulnerable_examples'] += 1
            elif not original_correct and not adversarial_correct:
                results['failed_examples'] += 1
            else:  # not original_correct and adversarial_correct
                results['improved_examples'] += 1
            
            # Store detailed results
            results['detailed_results'].append({
                'id': example['id'],
                'question': example['question'],
                'original_answer': example['original_answer'],
                'original_prediction': original_pred,
                'adversarial_prediction': adversarial_pred,
                'original_correct': original_correct,
                'adversarial_correct': adversarial_correct,
                'is_successful_attack': example['is_successful_attack'],
                'top_tokens': example['top_tokens']
            })
            
        except Exception as e:
            print(f"Error evaluating example {example['id']}: {e}")
            # Count as failed
            results['failed_examples'] += 1
            results['detailed_results'].append({
                'id': example['id'],
                'question': example['question'],
                'original_answer': example['original_answer'],
                'original_prediction': '',
                'adversarial_prediction': '',
                'original_correct': False,
                'adversarial_correct': False,
                'is_successful_attack': example['is_successful_attack'],
                'top_tokens': example['top_tokens'],
                'error': str(e)
            })
    
    return results

def is_answer_correct(prediction: str, ground_truth: str) -> bool:
    """Check if prediction is correct (simple string matching)."""
    if not ground_truth or not prediction:
        return ground_truth == prediction
    
    # Normalize strings for comparison - remove punctuation
    import re
    pred_norm = re.sub(r'[^\w\s]', '', prediction.lower().strip())
    gt_norm = re.sub(r'[^\w\s]', '', ground_truth.lower().strip())
    
    # Check for exact match or if prediction contains the ground truth
    return pred_norm == gt_norm or gt_norm in pred_norm

def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in a nice format."""
    total = results['total_examples']
    
    print("\n" + "="*80)
    print("GPT-2 VICTIM MODEL EVALUATION ON ADVERSARIAL DATASET")
    print("="*80)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Examples: {total}")
    print(f"   Correct on Original: {results['correct_on_original']} ({results['correct_on_original']/total:.2%})")
    print(f"   Correct on Adversarial: {results['correct_on_adversarial']} ({results['correct_on_adversarial']/total:.2%})")
    
    print(f"\n🛡️ ROBUSTNESS ANALYSIS:")
    print(f"   Robust Examples: {results['robust_examples']} ({results['robust_examples']/total:.2%})")
    print(f"   Vulnerable Examples: {results['vulnerable_examples']} ({results['vulnerable_examples']/total:.2%})")
    print(f"   Failed Examples: {results['failed_examples']} ({results['failed_examples']/total:.2%})")
    print(f"   Improved Examples: {results['improved_examples']} ({results['improved_examples']/total:.2%})")
    
    # Calculate robustness score
    robustness_score = results['robust_examples'] / (results['robust_examples'] + results['vulnerable_examples']) if (results['robust_examples'] + results['vulnerable_examples']) > 0 else 0
    print(f"\n🎯 ROBUSTNESS SCORE: {robustness_score:.2%}")
    
    # Show some examples of vulnerable cases
    vulnerable_examples = [r for r in results['detailed_results'] if r['original_correct'] and not r['adversarial_correct']]
    if vulnerable_examples:
        print(f"\n⚠️ SAMPLE VULNERABLE EXAMPLES:")
        for i, example in enumerate(vulnerable_examples[:3]):
            print(f"\n   Example {i+1} (ID: {example['id']}):")
            print(f"   Question: {example['question']}")
            print(f"   Original Answer: '{example['original_answer']}'")
            print(f"   Original Prediction: '{example['original_prediction']}'")
            print(f"   Adversarial Prediction: '{example['adversarial_prediction']}'")
            print(f"   Top Attacked Tokens: {[t['token'] for t in example['top_tokens'][:3]]}")
    
    # Show some examples of robust cases
    robust_examples = [r for r in results['detailed_results'] if r['original_correct'] and r['adversarial_correct']]
    if robust_examples:
        print(f"\n✅ SAMPLE ROBUST EXAMPLES:")
        for i, example in enumerate(robust_examples[:3]):
            print(f"\n   Example {i+1} (ID: {example['id']}):")
            print(f"   Question: {example['question']}")
            print(f"   Original Answer: '{example['original_answer']}'")
            print(f"   Original Prediction: '{example['original_prediction']}'")
            print(f"   Adversarial Prediction: '{example['adversarial_prediction']}'")
            print(f"   Top Attacked Tokens: {[t['token'] for t in example['top_tokens'][:3]]}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 victim model on adversarial dataset')
    parser.add_argument('--model_path', type=str, 
                       default='/data/arpit/code/outputs/gpt2_qa_switch_precision-20251007-130553',
                       help='Path to GPT-2 model')
    parser.add_argument('--dataset_file', type=str, default='adversarial_squad_dataset.json',
                       help='Adversarial dataset file')
    parser.add_argument('--output_file', type=str, default='gpt2_victim_evaluation_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--max_examples', type=int, default=None,
                       help='Maximum number of examples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Load adversarial dataset
    print(f"Loading adversarial dataset: {args.dataset_file}")
    examples = load_adversarial_dataset(args.dataset_file)
    
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Limited to {len(examples)} examples for testing")
    
    # Load GPT-2 victim model
    print(f"Loading GPT-2 victim model from: {args.model_path}")
    victim_model = GPT2QAVictimModel(args.model_path)
    
    # Evaluate model
    print(f"Evaluating GPT-2 victim model on {len(examples)} examples...")
    results = evaluate_gpt2_victim_on_examples(victim_model, examples)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved evaluation results to: {args.output_file}")

if __name__ == "__main__":
    main()
