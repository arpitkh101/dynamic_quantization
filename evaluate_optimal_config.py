#!/usr/bin/env python3
"""
Evaluate the optimal mixed quantization configuration on full dataset.
Configuration:
Early Layers (0–3):
- attn.c_proj: 4-bit (robust to quantization)
- All other components (attn.c_attn, mlp.c_fc, mlp.c_proj): 6-bit (sensitive foundational layers benefit from higher precision)
Middle Layers (4–8):
- attn.c_proj, mlp.c_proj: 6-bit (moderate robustness)
- Remaining components (attn.c_attn, mlp.c_fc): 8-bit (high sensitivity observed, even at 6-bit)
Final Layers (9–11):
- All components: 4-bit (robust and redundant, safe for aggressive quantization)
"""

import json
import os
import sys
import torch
import logging
from tqdm import tqdm
import numpy as np

# Add the current directory to Python path
sys.path.append('/data/arpit/code')

from models.modeling_gpt2_quant import GPT2ForQuestionAnswering
from models.configuration_gpt2 import GPT2Config
from config_new import config

# Import HuggingFace components for QA evaluation
import datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
    set_seed,
)
from torch.utils.data import DataLoader

from utils import postprocess_qa_predictions
from training_utils import create_data_preprocessing_functions, create_post_processing_function, create_compute_metrics_function

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_optimal_quant_config():
    """Create the optimal quantization configuration."""
    logger.info("Creating optimal quantization configuration...")
    
    quant_config = {
        'a_bit': {},
        'w_bit': {}
    }
    
    # All possible layers (48 total)
    all_layers = []
    for layer_idx in range(12):  # 0-11
        for component in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']:
            all_layers.append(f'transformer.h.{layer_idx}.{component}')
    
    # Configuration:
    # Early Layers (0–3):
    # - attn.c_proj: 4-bit (robust to quantization)
    # - All other components (attn.c_attn, mlp.c_fc, mlp.c_proj): 6-bit (sensitive foundational layers benefit from higher precision)
    # Middle Layers (4–8):
    # - attn.c_proj, mlp.c_proj: 6-bit (moderate robustness)
    # - Remaining components (attn.c_attn, mlp.c_fc): 8-bit (high sensitivity observed, even at 6-bit)
    # Final Layers (9–11):
    # - All components: 4-bit (robust and redundant, safe for aggressive quantization)
    
    layers_4bit = [
        # Final Layers (9–11): All components 4-bit

        'transformer.h.10.attn.c_attn',
        'transformer.h.10.attn.c_proj',
        'transformer.h.10.mlp.c_fc',
        'transformer.h.10.mlp.c_proj',
        'transformer.h.11.attn.c_attn',
        'transformer.h.11.attn.c_proj',
        'transformer.h.11.mlp.c_fc',
        'transformer.h.11.mlp.c_proj',
        
        # Early Layers (0–3): attn.c_proj 4-bit
        'transformer.h.0.attn.c_proj',
        'transformer.h.1.attn.c_proj',
        'transformer.h.2.attn.c_proj',
        'transformer.h.3.attn.c_proj',

        'transformer.h.9.attn.c_attn',
        'transformer.h.9.attn.c_proj',
        'transformer.h.9.mlp.c_fc',
        'transformer.h.9.mlp.c_proj',
        
    ]
    
    layers_6bit = [
        # Early Layers (0–3): All other components 6-bit
        'transformer.h.0.attn.c_attn',
        'transformer.h.0.mlp.c_fc',
        'transformer.h.0.mlp.c_proj',
        'transformer.h.1.attn.c_attn',
        'transformer.h.1.mlp.c_fc',
        'transformer.h.1.mlp.c_proj',
        'transformer.h.2.attn.c_attn',
        'transformer.h.2.mlp.c_fc',
        'transformer.h.2.mlp.c_proj',
        'transformer.h.3.attn.c_attn',
        'transformer.h.3.mlp.c_fc',
        'transformer.h.3.mlp.c_proj',
        
        # Middle Layers (4–8): attn.c_proj, mlp.c_proj 6-bit
        'transformer.h.4.attn.c_proj',   
        'transformer.h.4.mlp.c_fc',
        'transformer.h.5.attn.c_proj',    
        'transformer.h.5.mlp.c_fc',
        'transformer.h.6.attn.c_proj',
        'transformer.h.6.mlp.c_fc',
        'transformer.h.7.attn.c_proj',
        'transformer.h.7.mlp.c_fc',
        'transformer.h.8.attn.c_proj',
        'transformer.h.8.mlp.c_fc',
    ]
    
    layers_8bit = [
        # Middle Layers (4–8): attn.c_attn, mlp.c_fc 8-bit (high sensitivity)
        'transformer.h.4.attn.c_attn',
        'transformer.h.4.mlp.c_proj',
        'transformer.h.5.attn.c_attn',
        'transformer.h.5.mlp.c_proj',
        'transformer.h.6.attn.c_attn',
        'transformer.h.6.mlp.c_proj',
        'transformer.h.7.attn.c_attn',
        'transformer.h.7.mlp.c_proj',
        'transformer.h.8.attn.c_attn',
        'transformer.h.8.mlp.c_proj',
    ]
    
    for layer_name in all_layers:
        if layer_name in layers_4bit:
            quant_config['a_bit'][layer_name] = 4
            quant_config['w_bit'][layer_name] = 4
        elif layer_name in layers_6bit:
            quant_config['a_bit'][layer_name] = 6
            quant_config['w_bit'][layer_name] = 6
        elif layer_name in layers_8bit:
            quant_config['a_bit'][layer_name] = 8
            quant_config['w_bit'][layer_name] = 8
        else:
            # Default to 8-bit for any remaining layers
            quant_config['a_bit'][layer_name] = 8
            quant_config['w_bit'][layer_name] = 8
    
    # Count layers by bit width
    count_4bit = sum(1 for v in quant_config['a_bit'].values() if v == 4)
    count_6bit = sum(1 for v in quant_config['a_bit'].values() if v == 6)
    count_8bit = sum(1 for v in quant_config['a_bit'].values() if v == 8)
    
    logger.info(f"Optimal configuration:")
    logger.info(f"  4-bit layers: {count_4bit}")
    logger.info(f"  6-bit layers: {count_6bit}")
    logger.info(f"  8-bit layers: {count_8bit}")
    logger.info(f"  Total layers: {count_4bit + count_6bit + count_8bit}")
    
    return quant_config

def evaluate_model_optimal(model, eval_loader, eval_examples, eval_dataset, post_processing_function, compute_metrics, quant_config):
    """Evaluate the model with optimal quantization configuration."""
    logger.info("Starting model evaluation with optimal configuration...")
    
    model.eval()
    device = next(model.parameters()).device
    ignore_keys = ["loss"]  # Keys to ignore when extracting logits
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating Optimal Configuration"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Remove offset_mapping from batch before passing to model (it's only needed for post-processing)
            model_batch = {k: v for k, v in batch.items() if k != 'offset_mapping'}
            
            # Forward pass with quantization configuration
            outputs = model(**model_batch, quant_config=quant_config)
            
            # Extract logits exactly like custom_trainer.py
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            
            # Detach and move to CPU (like custom_trainer.py)
            logits = tuple(tensor.detach().cpu() for tensor in logits) if isinstance(logits, tuple) else logits.detach().cpu()
            
            if len(logits) == 1:
                logits = logits[0]
            
            all_predictions.append(logits)
    

    if all_predictions:
        if isinstance(all_predictions[0], tuple):
            # Handle tuple of logits (start_logits, end_logits)
            predictions = tuple(np.concatenate([pred[i].numpy() for pred in all_predictions], axis=0) 
                              for i in range(len(all_predictions[0])))
        else:
            # Handle single logits tensor
            predictions = np.concatenate([pred.numpy() for pred in all_predictions], axis=0)
    else:
        predictions = None
    

    if post_processing_function is not None and compute_metrics is not None:
        eval_preds = post_processing_function(eval_examples, eval_dataset, predictions)
        metrics = compute_metrics(eval_preds)
        
        logger.info("=" * 60)
        logger.info("OPTIMAL CONFIGURATION EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Exact Match: {metrics['exact_match']:.2f}")
        logger.info(f"F1 Score: {metrics['f1']:.2f}")
        logger.info("=" * 60)
        
        return metrics
    else:
        logger.error("Post-processing function or compute metrics not available")
        return None

def main():
    """Main evaluation function."""
    logger.info("Starting optimal quantization configuration evaluation on full dataset...")
    
    # Set random seed
    set_seed(config.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load full dataset
    logger.info("Loading full evaluation dataset...")
    raw_datasets = load_dataset("squad", split="validation")
    logger.info(f"Evaluating on {len(raw_datasets)} samples")
    
    # Create data args and column names
    class DataArgs:
        def __init__(self, config):
            self.max_seq_length = config.max_seq_length
            self.doc_stride = config.doc_stride
            self.pad_to_max_length = config.pad_to_max_length
            self.version_2_with_negative = config.version_2_with_negative
            self.n_best_size = config.n_best_size
            self.max_answer_length = config.max_answer_length
            self.null_score_diff_threshold = config.null_score_diff_threshold
    
    data_args = DataArgs(config)
    
    # Get column names
    column_names = raw_datasets.column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    
    # Create preprocessing functions
    prepare_train_features, prepare_eval_features = create_data_preprocessing_functions(
        tokenizer, data_args, question_column_name, context_column_name, answer_column_name
    )
    
    # Preprocess evaluation dataset
    eval_dataset = raw_datasets.map(
        prepare_eval_features,
        batched=True,
        remove_columns=raw_datasets.column_names,
        desc="Running tokenizer on evaluation dataset",
    )
    
    # Create custom QA data collator to handle None values (from layer_sensitivity.py)
    class QADataCollator:
        def __init__(self, tokenizer, pad_to_multiple_of=None):
            self.tokenizer = tokenizer
            self.pad_to_multiple_of = pad_to_multiple_of
        
        def __call__(self, features):
            # Handle offset_mapping separately to avoid None value issues
            offset_mappings = []
            for feature in features:
                if 'offset_mapping' in feature:
                    # Convert None values to (0, 0) for tensor creation
                    offset_mapping = []
                    for offset in feature['offset_mapping']:
                        if offset is None:
                            offset_mapping.append((0, 0))
                        else:
                            offset_mapping.append(offset)
                    offset_mappings.append(offset_mapping)
                    # Remove from feature to avoid conflicts with model forward
                    del feature['offset_mapping']
            
            # Use default data collator for the rest
            batch = default_data_collator(features)
            
            # Add offset_mappings back if they exist (for post-processing)
            if offset_mappings:
                batch['offset_mapping'] = offset_mappings
            
            return batch
    
    data_collator = QADataCollator(tokenizer, pad_to_multiple_of=8 if config.fp16 else None)
    
    # Create post-processing and compute metrics functions using training_utils (from layer_sensitivity.py)
    from training_utils import create_post_processing_function, create_compute_metrics_function
    
    # Create training args object for compatibility
    class TrainingArgs:
        def __init__(self, config):
            self.output_dir = config.output_dir
            self.fp16 = config.fp16
    
    training_args = TrainingArgs(config)
    
    # Create post-processing function (correct signature from layer_sensitivity.py)
    post_processing_function = create_post_processing_function(
        data_args, training_args, answer_column_name, logging.INFO
    )
    
    # Create compute metrics function
    class ModelArgs:
        def __init__(self, config):
            self.cache_dir = config.cache_dir
    
    model_args = ModelArgs(config)
    compute_metrics = create_compute_metrics_function(data_args, model_args)
    
    # Load model
    logger.info("Loading newer checkpoint model...")
    gpt2_config = GPT2Config.from_pretrained('gpt2')
    gpt2_config.a_bits = [4, 6, 8, 16]  # Include 6-bit for the 6-bit trained checkpoint
    gpt2_config.w_bits = [4, 6, 8, 16]  # Include 6-bit for the 6-bit trained checkpoint
    gpt2_config.ranks = [16, 8, 8, 8]   # 4-bit uses rank 16, others use rank 8 to match newer checkpoint
    gpt2_config.alphas = [32.0, 16.0, 16.0, 16.0]  # Adjust alphas for 6-bit
    gpt2_config.dropouts = [0.02, 0.01, 0.01, 0.01]  # Adjust dropouts for 6-bit
    gpt2_config.kv_bits = 16
    
    model = GPT2ForQuestionAnswering(gpt2_config)
    
    # Load pretrained weights
    checkpoint_path = '/data/arpit/code/outputs/gpt2_qa_switch_precision-20251007-130553/final_model.pt'
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    
    # Set layer names (from layer_sensitivity.py)
    for name, module in model.named_modules():
        if hasattr(module, 'set_layer_name'):
            clean_name = name.replace('module.', '') if name.startswith('module.') else name
            module.set_layer_name(clean_name)
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Create evaluation data loader (from layer_sensitivity.py)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config.dataloader_num_workers,
    )
    
    # Create optimal configuration
    quant_config = create_optimal_quant_config()
    
    # Evaluate model
    results = evaluate_model_optimal(
        model, eval_loader, raw_datasets, eval_dataset, 
        post_processing_function, compute_metrics, quant_config
    )
    
    if results:
        # Save results
        output_file = '/data/arpit/code/layer_sensitivity_results-20251007-010836/optimal_config_full_evaluation.json'
        with open(output_file, 'w') as f:
            json.dump({
                'exact_match': results['exact_match'],
                'f1': results['f1'],
                'eval_samples': len(eval_dataset),
                'quantization_config': quant_config,
                'configuration_description': {
                    'early_layers_0_3': 'attn.c_proj: 4-bit, others: 6-bit',
                    'middle_layers_4_8': 'attn.c_proj+mlp.c_proj: 6-bit, attn.c_attn+mlp.c_fc: 8-bit',
                    'final_layers_9_11': 'All components: 4-bit',
                    'strategy': 'Component-aware quantization based on sensitivity analysis'
                }
            }, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print comparison with baseline
        baseline_f1 = 73.75  # From newer checkpoint baseline
        baseline_em = 63.04
        
        f1_drop = ((baseline_f1 - results['f1']) / baseline_f1) * 100
        em_drop = ((baseline_em - results['exact_match']) / baseline_em) * 100
        
        logger.info(f"\n{'='*80}")
        logger.info("PERFORMANCE COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"Baseline F1: {baseline_f1:.2f}")
        logger.info(f"Optimal Config F1: {results['f1']:.2f} ({f1_drop:.1f}% drop)")
        logger.info(f"Baseline EM: {baseline_em:.2f}")
        logger.info(f"Optimal Config EM: {results['exact_match']:.2f} ({em_drop:.1f}% drop)")
        logger.info(f"{'='*80}")
        
        return results
    else:
        logger.error("Evaluation failed")
        return None

if __name__ == "__main__":
    results = main()
