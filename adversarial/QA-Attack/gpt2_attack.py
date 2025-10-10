#!/usr/bin/env python3
"""
GPT-2 QA-Attack Script
Adapted to work with the trained quantized GPT-2 QA model
"""

import sys
import os
sys.path.append('/data/arpit/code/adversarial/QA-Attack')

from gpt2_victim_model import GPT2QAVictimModel
from data import DatasetLoader, load_questions_and_contexts_from_json, fetch_document_content
import argparse
from transformers import BertTokenizer, BertForMaskedLM
import torch
from bert_mlm import BertMLMGuesser
from get_candidate import mask_and_predict, generate_candidate_sentences, count_unique_words_in_text1
from gpt2_ranking import ranking_with_attention_gpt2
from gpt2_tool import save_show_result_to_file, convert_seconds_to_hms
from gpt2_ranking import ranking_with_removal_gpt2
import time
import json
from gpt2_ranking import combine_ranking_scores_gpt2
import warnings

warnings.filterwarnings("ignore")

def attack_gpt2(args):
    """Attack function adapted for GPT-2 model"""
    if args.c == 1:
        args.single = True
    
    # Initialize GPT-2 Victim Model
    gpt2_victim_model = GPT2QAVictimModel(model_path=args.model_path)
    
    loader = DatasetLoader(args.dataset_name)
    loader.load_dataset()
    
    if args.dataset_name == 'google/boolq':
        formatted_strings = loader.get_formatted_string(split='validation')
    if args.dataset_name == 'deepmind/narrativeqa':
        formatted_strings = loader.get_formatted_string(split='test')
    
    data = loader.get_samples(num_samples=args.n, randomize=False)
    successful_attacks, ave_words = 0, 0
    raw_answers, all_attacked_answers, all_adversary, sentences, questions, best_candidates, results = [], [], [], [], [], [], []
    
    # Create results and cache directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    
    candidate_file = f'results/gpt2_candidate_{args.n}sample_{args.k}words_{args.c}_{args.ranking}_{args.combination}.txt'
    text_file = f'results/gpt2_{args.n}samples_attack{args.k}_words_{args.c}_{args.ranking}_{args.combination}.txt'
    
    print(f'\n -----------   Start GPT-2 attack with {args.ranking}   -------------')
    
    for i in range(args.n):
        print(f'\n ----- Attacking No.{i + 1} sample -----')
        
        if args.dataset_name == 'google/boolq':
            question, sentence = data[i][1][0], data[i][1][1]
        elif args.dataset_name == 'deepmind/narrativeqa':
            question, sentence = data[i][1][0], data[i][1][1]
            print(f'question: {question}')
            print(f'sentence: {sentence}')
        elif args.dataset_name == 'rajpurkar/squad' or args.dataset_name == 'rajpurkar/squad_v2':
            question, sentence, answer = data[i][1][0], data[i][1][1], data[i][2]
        else:
            # Default case
            question, sentence = data[i][1][0], data[i][1][1]
            answer = "N/A"
        
        # Get raw answer from GPT-2 model
        raw_answer = gpt2_victim_model.get_answer(question, sentence, bit_width=8)
        raw_logits = gpt2_victim_model.get_logits(question, sentence, bit_width=8)
        
        sentences.append(sentence)
        questions.append(question)
        
        # Apply ranking strategy
        if args.ranking == 'attention':
            top_k_context_tokens = ranking_with_attention_gpt2(question, sentence, gpt2_victim_model, top_k=args.k, rate=args.rate, bit_width=8)
            word_to_attack = [(j[1], j[2]) for j in top_k_context_tokens]  # (position, word) - format expected by original attack script
        elif args.ranking == 'removal':
            top_k_context_tokens = ranking_with_removal_gpt2(question, sentence, gpt2_victim_model, top_k=args.k, rate=args.rate, bit_width=8)
            word_to_attack = [(j[1], j[2]) for j in top_k_context_tokens]  # (position, word) - format expected by original attack script
        elif args.ranking == 'combined':
            top_k_context_tokens = combine_ranking_scores_gpt2(question, sentence, gpt2_victim_model, combination=args.combination, top_k=args.k, rate=args.rate, bit_width=8)
            word_to_attack = [(j[1][0], j[1][2]) for j in top_k_context_tokens]  # (position, word) - format expected by original attack script
        else:
            raise ValueError("Wrong ranking strategy input.")
        
        if args.rate:
            ave_words += len(word_to_attack)
        
        print(f'\n Top {args.k}/{args.rate} context tokens of sample {i + 1}: {top_k_context_tokens}')

        # Generate candidate sentences
        candidates = mask_and_predict(sentence, word_to_attack, num_of_predict=args.c, single=args.single)
        candidate_sentences = generate_candidate_sentences(sentence, word_to_attack, candidates, single=args.single)
        
        print(f'\n sentences: {sentence}')
        print(f'\n candidate_sentences: {candidate_sentences[0]}')
        
        attacked_answers = []
        max_logit_change = float('-inf')
        best_candidate = None
        successful_attack_flag = False
        
        for candidate_sentence in candidate_sentences:
            attacked_answer = gpt2_victim_model.get_answer(question, candidate_sentence, bit_width=8)
            candidate_logits = gpt2_victim_model.get_logits(question, candidate_sentence, bit_width=8)
            # Simple logit comparison - just use the max logit value difference
            logit_change = abs(torch.max(raw_logits).item() - torch.max(candidate_logits).item())
            attacked_answers.append(attacked_answer)

            # Update best candidate if current logit change is greater
            if logit_change > max_logit_change:
                max_logit_change = logit_change
                best_candidate = candidate_sentence

            # Check if the answer changed and no successful attack has been counted yet
            if attacked_answer != raw_answer and not successful_attack_flag:
                successful_attacks += 1
                successful_attack_flag = True
            
            # Break if a successful attack is found
            if successful_attack_flag:
                break

        if best_candidate:
            best_candidates.append(best_candidate)
        
        results.append({
            "index": i,
            "question": question,
            "answer": answer if 'answer' in locals() else "N/A",
            "raw_sentence": sentence,
            "best_candidate": best_candidate,
            "raw_answer": raw_answer,
            "predicted_answer": attacked_answers[0] if attacked_answers else "N/A"
        })
        
        print(f'\n raw_answer: {raw_answer}; attacked_answers: {attacked_answers}')
        raw_answers.append(raw_answer)
        all_attacked_answers.append(attacked_answers)
        all_adversary.append(candidate_sentences[0])
    
    # Save results
    with open(f'results/gpt2_{args.dataset_name.replace("/", "_")}_{args.n}samples_attack{args.k}_words_{args.c}_{args.ranking}_{args.combination}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    success_rate = successful_attacks / args.n
    show_result = [[ques, sent, adv, ans, raw] for ques, sent, adv, ans, raw in zip(questions, sentences, all_adversary, all_attacked_answers, raw_answers)]
    save_show_result_to_file(show_result, text_file)
    
    print(f'\n Successful attacks: {successful_attacks}, Total samples: {args.n}, Success rate: {success_rate:.2f}, Average words attack: {(ave_words/args.n):.2f}')
    return 'Finished!'

if __name__ == "__main__":
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="GPT-2 QA-Attack Parameters")
    parser.add_argument('--dataset_name', type=str, default='rajpurkar/squad', required=False, help='dataset name')
    parser.add_argument('--k', type=int, default=5, required=False, help='top k words been attacked')
    parser.add_argument('--c', type=int, default=2, required=False, help='number of candidates predicted from bert mlm')
    parser.add_argument('--n', type=int, default=2, required=False, help='number of test/validation data attacked')
    parser.add_argument('--ranking', type=str, default='combined', required=False, choices=['combined', 'attention', 'removal'], help='ranking strategy')
    parser.add_argument('--rate', type=float, default=None, required=False, help='percentage of words been attacked')
    parser.add_argument('--mode', type=str, default='yn', required=False, help='attacking mode regarding to question & answer type')    
    parser.add_argument('--single', type=str, default=False, required=False, help='only choose one candidate from BERT MLM')
    parser.add_argument('--combination', type=str, default='norm-link', required=False, choices=['norm-add', 'norm-link'], help='type of combination: norm-add, norm-link')
    parser.add_argument('--model_path', type=str, default="/data/arpit/code/outputs/gpt2_qa_switch_precision-20251004-125247", help='Path to the trained GPT-2 QA model')
    
    args = parser.parse_args()
    
    print(attack_gpt2(args))
    print(f'Arguments: {vars(args)}')
    
    end_time = time.time()
    total_time = end_time - start_time
    print(convert_seconds_to_hms(total_time))
