#!/usr/bin/env python3
"""
Evaluate the comprehensive adversarial dataset on both original and adversarial contexts.
Calculates F1 and EM metrics for comparison.
"""

import json
import sys
import os
import torch
import logging
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any, Tuple
import argparse
import random

# Add the parent directory to path to import our modules
sys.path.append('/data/arpit/code/')

from gpt2_victim_model import GPT2QAVictimModel
import evaluate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_comprehensive_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """Load the comprehensive adversarial dataset."""
    logger.info(f"Loading comprehensive adversarial dataset from {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    samples = []
    for article in data['data']:
        for qa in article['paragraphs'][0]['qas']:
            samples.append({
                'id': qa['id'],
                'question': qa['question'],
                'original_context': qa['context'],
                'adversarial_context': qa['adversarial_context'],
                'ground_truth_answers': qa['answers'],
                'original_model_answer': qa['original_model_answer'],
                'adversarial_answer': qa['adversarial_answer'],
                'attack_successful': qa['attack_successful'],
                'attack_method': qa['attack_method'],
                'attacked_tokens': qa['attacked_tokens']
            })
    
    logger.info(f"Loaded {len(samples)} samples from dataset")
    return samples

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (remove punctuation, lowercase, etc.)."""
    import re
    if not answer:
        return ""
    
    # Remove punctuation and convert to lowercase
    normalized = re.sub(r'[^\w\s]', '', answer.lower().strip())
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized

def calculate_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0
    
    # Calculate common tokens
    common_tokens = set(pred_tokens) & set(gt_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score between prediction and ground truth."""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)
    
    return 1.0 if pred_normalized == gt_normalized else 0.0

def evaluate_model_on_dataset(victim_model: GPT2QAVictimModel, samples: List[Dict[str, Any]], 
                            use_adversarial_context: bool = False, bit_width: int = 16) -> Dict[str, Any]:
    """Evaluate the model on the dataset using either original or adversarial contexts."""
    
    context_type = "adversarial" if use_adversarial_context else "original"
    logger.info(f"Evaluating model on {context_type} contexts...")
    
    results = {
        'total_samples': len(samples),
        'context_type': context_type,
        'f1_scores': [],
        'exact_matches': [],
        'predictions': [],
        'ground_truths': [],
        'sample_details': []
    }
    
    
    for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {context_type} contexts")):
        try:
            bit_width = random.choice([4,6,8,16])
            #bit_width = 4
            # Choose context based on evaluation type
            print(f"Bit width: {bit_width}")
            context = sample['adversarial_context'] if use_adversarial_context else sample['original_context']
            
            # Get model prediction
            prediction = victim_model.get_answer(sample['question'], context, bit_width=bit_width)
            
            # Get ground truth (use first answer as reference)
            ground_truth = sample['ground_truth_answers'][0]['text'] if sample['ground_truth_answers'] else ""
            
            # Calculate metrics
            f1_score = calculate_f1_score(prediction, ground_truth)
            exact_match = calculate_exact_match(prediction, ground_truth)
            
            # Store results
            results['f1_scores'].append(f1_score)
            results['exact_matches'].append(exact_match)
            results['predictions'].append(prediction)
            results['ground_truths'].append(ground_truth)
            
            # Store detailed sample information
            sample_detail = {
                'id': sample['id'],
                'question': sample['question'],
                'prediction': prediction,
                'ground_truth': ground_truth,
                'f1_score': f1_score,
                'exact_match': exact_match,
                'attack_successful': sample['attack_successful'],
                'attack_method': sample['attack_method']
            }
            results['sample_details'].append(sample_detail)
            
        except Exception as e:
            logger.error(f"Error evaluating sample {sample['id']}: {e}")
            # Add zero scores for failed evaluations
            results['f1_scores'].append(0.0)
            results['exact_matches'].append(0.0)
            results['predictions'].append("")
            results['ground_truths'].append("")
    
    # Calculate summary statistics
    results['avg_f1'] = np.mean(results['f1_scores'])
    results['avg_exact_match'] = np.mean(results['exact_matches'])
    results['total_f1'] = sum(results['f1_scores'])
    results['total_exact_match'] = sum(results['exact_matches'])
    
    return results

def analyze_attack_impact(original_results: Dict[str, Any], adversarial_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the impact of adversarial attacks on model performance."""
    
    logger.info("Analyzing attack impact...")
    
    analysis = {
        'overall_impact': {
            'f1_drop': original_results['avg_f1'] - adversarial_results['avg_f1'],
            'em_drop': original_results['avg_exact_match'] - adversarial_results['avg_exact_match'],
            'f1_drop_percentage': ((original_results['avg_f1'] - adversarial_results['avg_f1']) / original_results['avg_f1']) * 100 if original_results['avg_f1'] > 0 else 0,
            'em_drop_percentage': ((original_results['avg_exact_match'] - adversarial_results['avg_exact_match']) / original_results['avg_exact_match']) * 100 if original_results['avg_exact_match'] > 0 else 0
        },
        'successful_attacks_impact': {
            'f1_drop': 0.0,
            'em_drop': 0.0,
            'count': 0
        },
        'failed_attacks_impact': {
            'f1_drop': 0.0,
            'em_drop': 0.0,
            'count': 0
        },
        'attack_method_analysis': {}
    }
    
    # Analyze by attack success
    successful_attacks = []
    failed_attacks = []
    
    for i, sample in enumerate(original_results['sample_details']):
        if sample['attack_successful']:
            successful_attacks.append(i)
        else:
            failed_attacks.append(i)
    
    # Calculate impact for successful attacks
    if successful_attacks:
        orig_f1_successful = np.mean([original_results['f1_scores'][i] for i in successful_attacks])
        adv_f1_successful = np.mean([adversarial_results['f1_scores'][i] for i in successful_attacks])
        orig_em_successful = np.mean([original_results['exact_matches'][i] for i in successful_attacks])
        adv_em_successful = np.mean([adversarial_results['exact_matches'][i] for i in successful_attacks])
        
        analysis['successful_attacks_impact'] = {
            'f1_drop': orig_f1_successful - adv_f1_successful,
            'em_drop': orig_em_successful - adv_em_successful,
            'count': len(successful_attacks)
        }
    
    # Calculate impact for failed attacks
    if failed_attacks:
        orig_f1_failed = np.mean([original_results['f1_scores'][i] for i in failed_attacks])
        adv_f1_failed = np.mean([adversarial_results['f1_scores'][i] for i in failed_attacks])
        orig_em_failed = np.mean([original_results['exact_matches'][i] for i in failed_attacks])
        adv_em_failed = np.mean([adversarial_results['exact_matches'][i] for i in failed_attacks])
        
        analysis['failed_attacks_impact'] = {
            'f1_drop': orig_f1_failed - adv_f1_failed,
            'em_drop': orig_em_failed - adv_em_failed,
            'count': len(failed_attacks)
        }
    
    # Analyze by attack method
    attack_methods = set(sample['attack_method'] for sample in original_results['sample_details'])
    for method in attack_methods:
        method_indices = [i for i, sample in enumerate(original_results['sample_details']) 
                         if sample['attack_method'] == method]
        
        if method_indices:
            orig_f1_method = np.mean([original_results['f1_scores'][i] for i in method_indices])
            adv_f1_method = np.mean([adversarial_results['f1_scores'][i] for i in method_indices])
            orig_em_method = np.mean([original_results['exact_matches'][i] for i in method_indices])
            adv_em_method = np.mean([adversarial_results['exact_matches'][i] for i in method_indices])
            
            analysis['attack_method_analysis'][method] = {
                'count': len(method_indices),
                'f1_drop': orig_f1_method - adv_f1_method,
                'em_drop': orig_em_method - adv_em_method,
                'original_f1': orig_f1_method,
                'adversarial_f1': adv_f1_method,
                'original_em': orig_em_method,
                'adversarial_em': adv_em_method
            }
    
    return analysis

def print_evaluation_results(original_results: Dict[str, Any], adversarial_results: Dict[str, Any], 
                           analysis: Dict[str, Any]) -> None:
    """Print comprehensive evaluation results."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ADVERSARIAL DATASET EVALUATION RESULTS")
    print("="*80)
    
    # Overall performance comparison
    print(f"\n📊 OVERALL PERFORMANCE COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<20} {'Original':<15} {'Adversarial':<15} {'Drop':<15} {'Drop %':<15}")
    print("-" * 50)
    print(f"{'F1 Score':<20} {original_results['avg_f1']:<15.4f} {adversarial_results['avg_f1']:<15.4f} {analysis['overall_impact']['f1_drop']:<15.4f} {analysis['overall_impact']['f1_drop_percentage']:<15.2f}%")
    print(f"{'Exact Match':<20} {original_results['avg_exact_match']:<15.4f} {adversarial_results['avg_exact_match']:<15.4f} {analysis['overall_impact']['em_drop']:<15.4f} {analysis['overall_impact']['em_drop_percentage']:<15.2f}%")
    
    # Attack success analysis
    print(f"\n🎯 ATTACK SUCCESS ANALYSIS")
    print("-" * 50)
    successful_count = analysis['successful_attacks_impact']['count']
    failed_count = analysis['failed_attacks_impact']['count']
    total_count = successful_count + failed_count
    
    print(f"Successful attacks: {successful_count} ({successful_count/total_count*100:.1f}%)")
    print(f"Failed attacks: {failed_count} ({failed_count/total_count*100:.1f}%)")
    
    if successful_count > 0:
        print(f"\nSuccessful attacks impact:")
        print(f"  F1 drop: {analysis['successful_attacks_impact']['f1_drop']:.4f}")
        print(f"  EM drop: {analysis['successful_attacks_impact']['em_drop']:.4f}")
    
    if failed_count > 0:
        print(f"\nFailed attacks impact:")
        print(f"  F1 drop: {analysis['failed_attacks_impact']['f1_drop']:.4f}")
        print(f"  EM drop: {analysis['failed_attacks_impact']['em_drop']:.4f}")
    
    # Attack method analysis
    print(f"\n⚔️ ATTACK METHOD ANALYSIS")
    print("-" * 50)
    print(f"{'Method':<20} {'Count':<10} {'F1 Drop':<12} {'EM Drop':<12} {'Orig F1':<12} {'Adv F1':<12}")
    print("-" * 50)
    
    for method, stats in analysis['attack_method_analysis'].items():
        print(f"{method:<20} {stats['count']:<10} {stats['f1_drop']:<12.4f} {stats['em_drop']:<12.4f} {stats['original_f1']:<12.4f} {stats['adversarial_f1']:<12.4f}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 50)
    print(f"Total samples evaluated: {original_results['total_samples']}")
    print(f"Average F1 on original contexts: {original_results['avg_f1']:.4f}")
    print(f"Average F1 on adversarial contexts: {adversarial_results['avg_f1']:.4f}")
    print(f"Average EM on original contexts: {original_results['avg_exact_match']:.4f}")
    print(f"Average EM on adversarial contexts: {adversarial_results['avg_exact_match']:.4f}")
    print(f"Overall F1 degradation: {analysis['overall_impact']['f1_drop_percentage']:.2f}%")
    print(f"Overall EM degradation: {analysis['overall_impact']['em_drop_percentage']:.2f}%")

def save_evaluation_results(original_results: Dict[str, Any], adversarial_results: Dict[str, Any], 
                          analysis: Dict[str, Any], output_file: str) -> None:
    """Save evaluation results to JSON file."""
    
    results_summary = {
        'evaluation_summary': {
            'total_samples': original_results['total_samples'],
            'original_performance': {
                'avg_f1': original_results['avg_f1'],
                'avg_exact_match': original_results['avg_exact_match'],
                'total_f1': original_results['total_f1'],
                'total_exact_match': original_results['total_exact_match']
            },
            'adversarial_performance': {
                'avg_f1': adversarial_results['avg_f1'],
                'avg_exact_match': adversarial_results['avg_exact_match'],
                'total_f1': adversarial_results['total_f1'],
                'total_exact_match': adversarial_results['total_exact_match']
            },
            'overall_impact': analysis['overall_impact'],
            'attack_success_analysis': {
                'successful_attacks': analysis['successful_attacks_impact'],
                'failed_attacks': analysis['failed_attacks_impact']
            },
            'attack_method_analysis': analysis['attack_method_analysis']
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {output_file}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate comprehensive adversarial dataset')
    parser.add_argument('--dataset_file', type=str, default='comprehensive_adversarial_dataset_500.json',
                       help='Path to the comprehensive adversarial dataset')
    parser.add_argument('--model_path', type=str, 
                       default='/data/arpit/code/outputs/gpt2_qa_switch_precision-20251007-130553',
                       help='Path to the GPT-2 victim model')
    parser.add_argument('--bit_width', type=int, default=16,
                       help='Bit width for model evaluation')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    logger.info("Starting comprehensive adversarial dataset evaluation...")
    
    # Load GPT-2 victim model
    logger.info("Loading GPT-2 victim model...")
    try:
        victim_model = GPT2QAVictimModel(args.model_path)
        logger.info("✅ GPT-2 victim model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return
    
    # Load comprehensive dataset
    samples = load_comprehensive_dataset(args.dataset_file)
    
    # Evaluate on original contexts
    logger.info("Evaluating on original contexts...")
    original_results = evaluate_model_on_dataset(victim_model, samples, use_adversarial_context=False, bit_width=args.bit_width)
    
    # Evaluate on adversarial contexts
    logger.info("Evaluating on adversarial contexts...")
    adversarial_results = evaluate_model_on_dataset(victim_model, samples, use_adversarial_context=True, bit_width=args.bit_width)
    
    # Analyze attack impact
    analysis = analyze_attack_impact(original_results, adversarial_results)
    
    # Print results
    print_evaluation_results(original_results, adversarial_results, analysis)
    
    # Save results
    save_evaluation_results(original_results, adversarial_results, analysis, args.output_file)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
