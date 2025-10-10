#!/usr/bin/env python3
"""
Create a comprehensive adversarial dataset with hybrid attacks.
Each sample contains original fields, model answers, attack success status, and adversarial examples.
"""

import sys
import os
import json
import random
from typing import Dict, List, Any, Tuple
import argparse

# Add the parent directory to path to import our modules
sys.path.append('/data/arpit/code/')
from gpt2_victim_model import GPT2QAVictimModel
from gpt2_ranking import ranking_with_attention_gpt2, ranking_with_removal_gpt2, combine_ranking_scores_gpt2
from get_candidate import mask_and_predict, generate_candidate_sentences
from data import DatasetLoader

def load_squad_samples(num_samples: int = 500) -> List[Dict[str, Any]]:
    """Load SQuAD samples for adversarial generation."""
    print(f"Loading {num_samples} samples from SQuAD...")
    
    # Use the existing data loader
    loader = DatasetLoader('rajpurkar/squad')
    loader.load_dataset()  # Load the dataset first
    data = loader.get_samples(num_samples=num_samples, randomize=True)
    
    # Convert to list of dictionaries
    # Data format: (index, [question, context], [answers])
    samples = []
    for i in range(len(data)):
        sample = data[i]
        index, qa_pair, answers = sample
        question, context = qa_pair
        
        samples.append({
            'id': f'squad_{index}',
            'question': question,
            'context': context,
            'answers': [{'text': ans, 'answer_start': context.find(ans) if ans in context else 0} for ans in answers]
        })
    
    print(f"Loaded {len(samples)} samples from SQuAD")
    return samples

def apply_hybrid_attack(question: str, context: str, victim_model: GPT2QAVictimModel) -> Dict[str, Any]:
    """
    Apply hybrid attack (attention + removal) to a question-context pair.
    Returns comprehensive attack results.
    """
    print(f"    Applying hybrid attack...")
    
    # Get original model prediction
    original_answer = victim_model.get_answer(question, context, bit_width=4)
    print(f"    Original answer: '{original_answer}'")
    
    # Initialize result structure
    result = {
        'original_model_answer': original_answer,
        'attack_successful': False,
        'adversarial_context': context,  # Default to original if no attack
        'adversarial_answer': original_answer,  # Default to original if no attack
        'attack_method': 'hybrid',
        'attacked_tokens': [],
        'attack_details': {
            'attention_tokens': [],
            'removal_tokens': [],
            'hybrid_tokens': [],
            'candidates_tested': 0
        }
    }
    
    try:
        # Get attention-based ranking
        print(f"      Computing attention-based ranking...")
        attention_ranked_tokens = ranking_with_attention_gpt2(
            question, context, victim_model, 
            top_k=5, rate=0.1, bit_width=4
        )
        result['attack_details']['attention_tokens'] = [
            {'token': token, 'position': int(pos), 'score': float(score)} 
            for score, pos, token, _ in attention_ranked_tokens[:3]
        ]
        
        # Get removal-based ranking
        print(f"      Computing removal-based ranking...")
        removal_ranked_tokens = ranking_with_removal_gpt2(
            question, context, victim_model, 
            top_k=5, rate=0.1, bit_width=4
        )
        result['attack_details']['removal_tokens'] = [
            {'token': token, 'position': int(pos), 'score': float(score)} 
            for score, pos, token, _ in removal_ranked_tokens[:3]
        ]
        
        # Get hybrid ranking
        print(f"      Computing hybrid ranking...")
        hybrid_ranked_tokens = combine_ranking_scores_gpt2(
            question, context, victim_model, 
            combination='norm-link', top_k=5, rate=0.1, bit_width=4
        )
        result['attack_details']['hybrid_tokens'] = [
            {'token': token, 'position': int(pos), 'score': float(score)} 
            for score, (pos, token, _) in hybrid_ranked_tokens[:3]
        ]
        
        # Try attacks in order: attention, removal, hybrid
        attack_methods = [
            ('attention', attention_ranked_tokens),
            ('removal', removal_ranked_tokens),
            ('hybrid', hybrid_ranked_tokens)
        ]
        
        for method_name, ranked_tokens in attack_methods:
            print(f"      Trying {method_name}-based attacks...")
            
            if not ranked_tokens:
                continue
                
            # Test top 3 tokens from this method
            top_tokens = ranked_tokens[:3]
            
            for token_info in top_tokens:
                try:
                    # Handle different formats
                    if method_name == 'hybrid':
                        score, token_tuple = token_info
                        position, token, cleaned_token = token_tuple
                    else:
                        score, position, token, cleaned_token = token_info
                    
                    print(f"        Attacking token: '{token}' (score: {score:.4f})")
                    
                    # Generate predictions using BERT
                    predictions = mask_and_predict(context, [(position, token)], num_of_predict=5)
                    
                    # Generate candidate sentences
                    candidates = generate_candidate_sentences(context, [(position, token)], predictions)
                    
                    # Test each candidate
                    for candidate_context in candidates:
                        result['attack_details']['candidates_tested'] += 1
                        
                        try:
                            adversarial_answer = victim_model.get_answer(question, candidate_context, bit_width=4)
                            
                            # Check if attack was successful
                            is_successful = original_answer != adversarial_answer
                            
                            # Always update with the latest attempt
                            result['adversarial_context'] = candidate_context
                            result['adversarial_answer'] = adversarial_answer
                            result['attacked_tokens'].append({
                                'token': token,
                                'position': int(position),
                                'score': float(score),
                                'method': method_name,
                                'successful': is_successful
                            })
                            
                            if is_successful:
                                print(f"          🎉 SUCCESS! Changed from '{original_answer}' to '{adversarial_answer}'")
                                result['attack_successful'] = True
                                result['attack_method'] = f'hybrid-{method_name}'
                                return result
                            else:
                                print(f"          ❌ Failed - same prediction")
                                
                        except Exception as e:
                            print(f"          ❌ Error testing candidate: {e}")
                            continue
                    
                except Exception as e:
                    print(f"        ❌ Error attacking token: {e}")
                    continue
        
        print(f"    No successful attacks found")
        return result
        
    except Exception as e:
        print(f"    ❌ Error in hybrid attack: {e}")
        return result

def create_comprehensive_dataset(num_samples: int = 500, test_size: int = 10) -> None:
    """
    Create a comprehensive adversarial dataset with hybrid attacks.
    """
    print("="*80)
    print("CREATING COMPREHENSIVE ADVERSARIAL DATASET")
    print("="*80)
    
    # Load GPT-2 victim model
    print("\n🔧 Loading GPT-2 Victim Model")
    print("-" * 50)
    model_path = '/data/arpit/code/outputs/gpt2_qa_switch_precision-20251007-130553'
    
    try:
        victim_model = GPT2QAVictimModel(model_path)
        print("✅ GPT-2 victim model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load SQuAD samples
    print(f"\n📚 Loading {num_samples} SQuAD Samples")
    print("-" * 50)
    samples = load_squad_samples(num_samples)
    
    # Create comprehensive dataset
    print(f"\n⚔️ Generating Adversarial Examples")
    print("-" * 50)
    
    comprehensive_data = {
        'version': '2.0',
        'data': []
    }
    
    successful_attacks = 0
    total_attacks = 0
    
    for i, sample in enumerate(samples):
        print(f"\n📝 Processing Sample {i+1}/{len(samples)}")
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question'][:80]}...")
        
        # Apply hybrid attack
        attack_result = apply_hybrid_attack(
            sample['question'], 
            sample['context'], 
            victim_model
        )
        
        # Create comprehensive sample
        comprehensive_sample = {
            'id': sample['id'],
            'question': sample['question'],
            'context': sample['context'],
            'answers': sample['answers'],
            'is_impossible': False,
            'original_model_answer': attack_result['original_model_answer'],
            'attack_successful': attack_result['attack_successful'],
            'adversarial_context': attack_result['adversarial_context'],
            'adversarial_answer': attack_result['adversarial_answer'],
            'attack_method': attack_result['attack_method'],
            'attacked_tokens': attack_result['attacked_tokens'],
            'attack_details': attack_result['attack_details']
        }
        
        # Track statistics
        total_attacks += 1
        if attack_result['attack_successful']:
            successful_attacks += 1
        
        # Add to dataset
        article = {
            'title': f"Comprehensive Adversarial Sample - {sample['id']}",
            'paragraphs': [{
                'context': sample['context'],
                'qas': [comprehensive_sample]
            }]
        }
        comprehensive_data['data'].append(article)
        
        # Progress update
        if (i + 1) % 10 == 0:
            success_rate = (successful_attacks / total_attacks) * 100
            print(f"    Progress: {i+1}/{len(samples)} | Success rate: {success_rate:.1f}%")
    
    # Save dataset
    output_file = f'comprehensive_adversarial_dataset_{num_samples}.json'
    print(f"\n💾 Saving Dataset")
    print("-" * 50)
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_data, f, indent=2)
    
    print(f"✅ Dataset saved to: {output_file}")
    
    # Final statistics
    final_success_rate = (successful_attacks / total_attacks) * 100
    print(f"\n📊 Final Statistics")
    print("=" * 50)
    print(f"Total samples: {total_attacks}")
    print(f"Successful attacks: {successful_attacks}")
    print(f"Attack success rate: {final_success_rate:.2f}%")
    total_candidates = sum(article['paragraphs'][0]['qas'][0]['attack_details']['candidates_tested'] for article in comprehensive_data['data'])
    print(f"Average candidates tested per sample: {total_candidates / total_attacks:.1f}")
    
    # Test with small sample if requested
    if test_size > 0:
        print(f"\n🧪 Testing with {test_size} samples")
        print("-" * 50)
        test_samples = random.sample(samples, min(test_size, len(samples)))
        
        test_data = {
            'version': '2.0',
            'data': []
        }
        
        for i, sample in enumerate(test_samples):
            print(f"Testing sample {i+1}/{len(test_samples)}: {sample['id']}")
            attack_result = apply_hybrid_attack(
                sample['question'], 
                sample['context'], 
                victim_model
            )
            
            comprehensive_sample = {
                'id': sample['id'],
                'question': sample['question'],
                'context': sample['context'],
                'answers': sample['answers'],
                'is_impossible': False,
                'original_model_answer': attack_result['original_model_answer'],
                'attack_successful': attack_result['attack_successful'],
                'adversarial_context': attack_result['adversarial_context'],
                'adversarial_answer': attack_result['adversarial_answer'],
                'attack_method': attack_result['attack_method'],
                'attacked_tokens': attack_result['attacked_tokens'],
                'attack_details': attack_result['attack_details']
            }
            
            article = {
                'title': f"Test Sample - {sample['id']}",
                'paragraphs': [{
                    'context': sample['context'],
                    'qas': [comprehensive_sample]
                }]
            }
            test_data['data'].append(article)
        
        test_output_file = f'comprehensive_adversarial_dataset_test_{test_size}.json'
        with open(test_output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"✅ Test dataset saved to: {test_output_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create comprehensive adversarial dataset with hybrid attacks')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples to process')
    parser.add_argument('--test_size', type=int, default=10,
                       help='Number of samples for testing (0 to skip)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create dataset
    create_comprehensive_dataset(args.num_samples, args.test_size)

if __name__ == "__main__":
    main()
