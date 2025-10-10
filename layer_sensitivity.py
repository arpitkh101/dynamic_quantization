from __future__ import division
import os
import sys
import time
import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# Import our config
from config_new import config

# Import HuggingFace components for QA
import datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Import our custom components
from models.configuration_gpt2 import GPT2Config
from models.quantized_lora_linear import QuantizeLoraLinear
from utils import postprocess_qa_predictions
from training_utils import create_data_preprocessing_functions
import copy
from typing import List
from utils import QADistillationLoss
import argparse
from utils import (
    get_lora_params, count_trainable_parameters, freeze_model_for_lora_training,
    save_model, save_training_state, load_training_state, 
    save_metrics_to_json, save_tokenizer, load_model
)

parser = argparse.ArgumentParser(description='GPT-2 QA Training with Switch Precision')
# Dataset and model arguments
parser.add_argument('--dataset_path', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--dataset_name', type=str, default='squad',
                    help='dataset name')
parser.add_argument('--dataset_config_name', type=str, default=None,
                    help='dataset config name')
parser.add_argument('--model_name_or_path', type=str, default='gpt2',
                    help='path to pretrained model')
parser.add_argument('--config_name', type=str, default=None,
                    help='pretrained config name or path if not the same as model_name')
parser.add_argument('--tokenizer_name', type=str, default=None,
                    help='pretrained tokenizer name or path if not the same as model_name')
parser.add_argument('--cache_dir', type=str, default=None,
                    help='path to directory to store the pretrained models downloaded from huggingface.co')
parser.add_argument('--model_revision', type=str, default='main',
                    help='the specific model version to use (can be a branch name, tag name or commit id)')
parser.add_argument('--token', type=str, default=None,
                    help='the token to use as HTTP bearer authorization for remote files')
parser.add_argument('--trust_remote_code', action='store_true',
                    help='whether to trust the execution of code from datasets/models defined on the Hub')
parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='output directory')
parser.add_argument('--overwrite_output_dir', action='store_true',
                    help='overwrite output directory')
parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                    help='path to checkpoint to resume from')

# Evaluation arguments
parser.add_argument('--per_device_eval_batch_size', type=int, default=16,
                    help='batch size per device for evaluation')
parser.add_argument('--max_seq_length', type=int, default=786,
                    help='maximum sequence length')
parser.add_argument('--max_eval_samples', type=int, default=None,
                    help='maximum number of evaluation samples')

# Logging and evaluation arguments
parser.add_argument('--logging_strategy', type=str, default='steps',
                    help='logging strategy')
parser.add_argument('--logging_steps', type=int, default=50,
                    help='logging steps')
parser.add_argument('--eval_strategy', type=str, default='steps',
                    help='evaluation strategy')
parser.add_argument('--eval_steps', type=int, default=1000,
                    help='evaluation steps')
parser.add_argument('--save_strategy', type=str, default='steps',
                    help='save strategy')
parser.add_argument('--save_steps', type=int, default=1000,
                    help='save steps')

# Learning rate scheduler
parser.add_argument('--lr_scheduler_type', type=str, default='cosine',
                    help='learning rate scheduler type')

# Model loading options
parser.add_argument('--pretrained_model_path', type=str, required=True,
                    help='path to pretrained model checkpoint (.pt file)')
parser.add_argument('--use_from_pretrained', action='store_true',
                    help='use from_pretrained method instead of manual loading')
parser.add_argument('--layer_config_path', type=str, default=None,
                    help='path to layer-wise quantization configuration file (optional)')
parser.add_argument('--test_bits', nargs='+', type=int, default=[4, 6],
                    help='bit widths to test for each layer')

# Original training config parameters (needed for model initialization)
parser.add_argument('--original_a_bits', nargs='+', type=int, default=[4, 6, 8, 16],
                    help='original activation bits used during training')
parser.add_argument('--original_w_bits', nargs='+', type=int, default=[4, 6, 8, 16],
                    help='original weight bits used during training')
parser.add_argument('--original_ranks', nargs='+', type=int, default=[16, 16, 16, 16],
                    help='original LoRA ranks used during training')
parser.add_argument('--original_alphas', nargs='+', type=float, default=[32.0, 32.0, 32.0, 32.0],
                    help='original LoRA alphas used during training')
parser.add_argument('--original_dropouts', nargs='+', type=float, default=[0.02, 0.02, 0.02, 0.02],
                    help='original LoRA dropouts used during training')
parser.add_argument('--original_kv_bits', type=int, default=16,
                    help='original key-value bits used during training')

args = parser.parse_args()

def load_base_config(config_path):
    """Load base layer-wise quantization configuration from JSON file"""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_pretrained_model(model, checkpoint_path):
    """Load pretrained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle DataParallel case
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    logging.info(f"Loaded pretrained model from {checkpoint_path}")
    return model

def load_model_from_pretrained(model_name_or_path, original_config_params, checkpoint_path=None):
    """Load model using from_pretrained with custom config parameters"""
    from models.modeling_gpt2_quant import GPT2ForQuestionAnswering
    
    # Load base config from HuggingFace
    base_config = GPT2Config.from_pretrained(model_name_or_path)
    
    # Add custom quantization parameters
    base_config.a_bits = original_config_params['a_bits']
    base_config.w_bits = original_config_params['w_bits']
    base_config.ranks = original_config_params['ranks']
    base_config.alphas = original_config_params['alphas']
    base_config.dropouts = original_config_params['dropouts']
    base_config.kv_bits = original_config_params['kv_bits']
    
    # Create model with custom config
    model = GPT2ForQuestionAnswering(base_config)
    
    # Load pretrained weights if checkpoint is provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        logging.info(f"Loaded pretrained weights from {checkpoint_path}")
    else:
        # Load from HuggingFace (this won't work perfectly due to custom layers)
        logging.warning("Loading from HuggingFace with custom layers may not work correctly")
        # model = GPT2ForQuestionAnswering.from_pretrained(model_name_or_path, config=base_config)
    
    return model

def get_model_layers(model):
    """Extract layer names from model that support quantization"""
    layer_names = []
    for name, module in model.named_modules():
        # Check if this is a quantizable layer (has set_layer_name method like QuantizeLoraLinear)
        if hasattr(module, 'set_layer_name'):
            # Remove 'module.' prefix if present (DataParallel)
            clean_name = name.replace('module.', '') if name.startswith('module.') else name
            layer_names.append(clean_name)
    return layer_names


def generate_layer_sensitivity_configs(layer_names, test_bits=[4, 6]):
    """Generate configurations for layer-wise sensitivity analysis"""
    configs = {}
    
    # Create baseline config (all layers at 8 bits) - using singular form for model
    baseline_config = {
        'a_bit': {layer: 8 for layer in layer_names},
        'w_bit': {layer: 8 for layer in layer_names}
    }
    configs['baseline'] = baseline_config
    
    # For each layer, test different bit combinations (keeping a_bit = w_bit for simplicity)
    for layer_name in layer_names:
        for bit in test_bits:
            config_name = f"{layer_name}_{bit}bit"
            
            # Start with baseline
            config = {
                'a_bit': baseline_config['a_bit'].copy(),
                'w_bit': baseline_config['w_bit'].copy()
            }
            
            # Modify only the target layer (same bit for both activation and weight)
            config['a_bit'][layer_name] = bit
            config['w_bit'][layer_name] = bit
            
            configs[config_name] = config
    
    return configs

def set_layer_names(model):
    """Set layer names for QuantizeLoraLinear modules"""
    for name, module in model.named_modules():
        if hasattr(module, 'set_layer_name'):
            # Remove 'module.' prefix if present (DataParallel)
            clean_name = name.replace('module.', '') if name.startswith('module.') else name
            module.set_layer_name(clean_name)

def create_dataparallel_quant_config(quant_config):
    """Create quant_config that works with DataParallel models"""
    # DataParallel adds 'module.' prefix to layer names
    dataparallel_config = {
        'a_bit': {},
        'w_bit': {}
    }
    
    for layer_name, a_bit in quant_config['a_bit'].items():
        # Add both original and module-prefixed versions
        dataparallel_config['a_bit'][layer_name] = a_bit
        dataparallel_config['a_bit'][f'module.{layer_name}'] = a_bit
    
    for layer_name, w_bit in quant_config['w_bit'].items():
        # Add both original and module-prefixed versions
        dataparallel_config['w_bit'][layer_name] = w_bit
        dataparallel_config['w_bit'][f'module.{layer_name}'] = w_bit
    
    # Debug: Print the created config
    print(f"DEBUG: Created dataparallel_config with {len(dataparallel_config['a_bit'])} a_bit entries and {len(dataparallel_config['w_bit'])} w_bit entries")
    
    return dataparallel_config

def create_heatmaps(results, output_dir, test_bits=[4, 6]):
    """Create heatmaps for layer sensitivity analysis"""
    import seaborn as sns
    import pandas as pd
    
    # Extract layer names and organize results
    layer_results = {}
    
    for config_name, metrics in results.items():
        if config_name == 'baseline':
            continue
            
        # Parse config name to extract layer and bit
        # Format: "layer_name_{bit}bit"
        if config_name.endswith('bit'):
            parts = config_name.rsplit('_', 1)  # Split on last underscore
            if len(parts) == 2:
                layer_name = parts[0]
                bit_str = parts[1]  # e.g., "4bit"
                bit = int(bit_str[:-3])  # Remove "bit" suffix
                
                if layer_name not in layer_results:
                    layer_results[layer_name] = {}
                
                # Store F1 score (or other metric)
                f1_score = metrics.get('f1', 0.0)
                layer_results[layer_name][bit] = f1_score
    
    # Create bar charts for each layer
    for layer_name, bit_results in layer_results.items():
        # Create data for bar chart
        bit_values = []
        f1_scores = []
        
        for bit in test_bits:
            bit_values.append(f"{bit}-bit")
            f1_scores.append(bit_results.get(bit, 0.0))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Bit Width': bit_values,
            'F1 Score': f1_scores
        })
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Bit Width'], df['F1 Score'], 
                      color=['red' if x < max(f1_scores) else 'green' for x in f1_scores])
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.title(f'Layer Sensitivity: {layer_name}')
        plt.xlabel('Bit Width (a_bit = w_bit)')
        plt.ylabel('F1 Score')
        plt.ylim(0, max(f1_scores) * 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
        plt.savefig(os.path.join(output_dir, f'heatmap_{safe_layer_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chart for {layer_name}")
    
    # Create summary bar chart (average across all layers)
    if layer_results:
        summary_data = {bit: [] for bit in test_bits}
        
        for layer_name, bit_results in layer_results.items():
            for bit in test_bits:
                score = bit_results.get(bit, 0.0)
                summary_data[bit].append(score)
        
        # Calculate averages
        avg_scores = []
        bit_labels = []
        for bit in test_bits:
            scores = summary_data[bit]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            avg_scores.append(avg_score)
            bit_labels.append(f"{bit}-bit")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Bit Width': bit_labels,
            'Average F1 Score': avg_scores
        })
        
        # Create summary bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.bar(summary_df['Bit Width'], summary_df['Average F1 Score'],
                      color=['red' if x < max(avg_scores) else 'green' for x in avg_scores])
        
        # Add value labels on bars
        for bar, score in zip(bars, avg_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.2f}', ha='center', va='bottom')
        
        plt.title('Layer Sensitivity Summary (Average F1 Score across all layers)')
        plt.xlabel('Bit Width (a_bit = w_bit)')
        plt.ylabel('Average F1 Score')
        plt.ylim(0, max(avg_scores) * 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save summary chart
        plt.savefig(os.path.join(output_dir, 'heatmap_summary.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved summary chart")

def create_layer_chart(layer_name, bit_results, test_bits, output_dir):
    """Create and save a chart for a specific layer"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create data for bar chart
    bit_values = []
    f1_scores = []
    
    for bit in test_bits:
        bit_values.append(f"{bit}-bit")
        f1_scores.append(bit_results.get(bit, 0.0))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Bit Width': bit_values,
        'F1 Score': f1_scores
    })
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Bit Width'], df['F1 Score'], 
                  color=['red' if x < max(f1_scores) else 'green' for x in f1_scores])
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.title(f'Layer Sensitivity: {layer_name}')
    plt.xlabel('Bit Width (a_bit = w_bit)')
    plt.ylabel('F1 Score')
    plt.ylim(0, max(f1_scores) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save chart
    safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
    chart_path = os.path.join(output_dir, f'layer_chart_{safe_layer_name}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved chart for {layer_name}: {chart_path}")

def main():
    # Update config from arguments
    from config_new import update_config_from_args
    update_config_from_args(args)
    
    # Setup logging and output directory
    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    config.output_dir = config.save
    

    
    os.makedirs(config.save, exist_ok=True)
    
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("args = %s", str(config))
    
    # Set seed
    set_seed(config.seed)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name if config.tokenizer_name else config.model_name_or_path,
        cache_dir=config.cache_dir,
        use_fast=True,
        revision=config.model_revision,
        token=config.token,
        trust_remote_code=config.trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load custom config
    gpt2_config = GPT2Config.from_pretrained(
        config.config_name if config.config_name else config.model_name_or_path,
        cache_dir=config.cache_dir,
        revision=config.model_revision,
        token=config.token,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Add required quantization parameters for model initialization
    # Use the original training config parameters to match the pretrained model
    gpt2_config.a_bits = args.original_a_bits
    gpt2_config.w_bits = args.original_w_bits
    gpt2_config.ranks = args.original_ranks
    gpt2_config.alphas = args.original_alphas
    gpt2_config.dropouts = args.original_dropouts
    gpt2_config.kv_bits = args.original_kv_bits
    
    logging.info(f"Using original training config: a_bits={gpt2_config.a_bits}, w_bits={gpt2_config.w_bits}")
    logging.info(f"LoRA config: ranks={gpt2_config.ranks}, alphas={gpt2_config.alphas}, dropouts={gpt2_config.dropouts}")

    # Load pretrained model from checkpoint
    if args.use_from_pretrained:
        # Use from_pretrained style loading
        original_config_params = {
            'a_bits': args.original_a_bits,
            'w_bits': args.original_w_bits,
            'ranks': args.original_ranks,
            'alphas': args.original_alphas,
            'dropouts': args.original_dropouts,
            'kv_bits': args.original_kv_bits
        }
        model = load_model_from_pretrained(
            config.model_name_or_path, 
            original_config_params, 
            args.pretrained_model_path
        )
        logging.info("Loaded pretrained model using from_pretrained style")
    else:
        # Use manual loading (original method)
        model = load_model(gpt2_config)
        model = load_pretrained_model(model, args.pretrained_model_path)
        logging.info("Loaded pretrained model from checkpoint")
    
    # Set layer names for QuantizeLoraLinear modules
    set_layer_names(model)
    
    # Move model to GPU with DataParallel support
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            # Multiple GPUs available - use DataParallel
            model = torch.nn.DataParallel(model).cuda()
            logging.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        else:
            # Single GPU - just move to GPU
            model = model.cuda()
            logging.info("Using single GPU")
    else:
        # No GPU available - keep on CPU
        logging.warning("CUDA not available, using CPU")

    # Load dataset
    if config.dataset_name is not None:
        raw_datasets = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            cache_dir=config.cache_dir,
            token=config.token,
            trust_remote_code=config.trust_remote_code,
        )
    else:
        # Handle custom dataset loading
        data_files = {}
        if hasattr(config, 'train_file') and config.train_file:
            data_files["train"] = config.train_file
        if hasattr(config, 'validation_file') and config.validation_file:
            data_files["validation"] = config.validation_file
        
        extension = "json"  # Default to JSON
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=config.cache_dir,
            token=config.token,
            trust_remote_code=config.trust_remote_code,
        )

    # Get column names for evaluation
    column_names = raw_datasets["validation"].column_names
    
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Create data preprocessing functions
    # Create a data_args object with the necessary attributes
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
    prepare_train_features, prepare_validation_features = create_data_preprocessing_functions(
        tokenizer, data_args, question_column_name, context_column_name, answer_column_name
    )

    # Prepare evaluation dataset
    eval_examples = raw_datasets["validation"]
    if config.max_eval_samples is not None:
        max_eval_samples = min(len(eval_examples), config.max_eval_samples)
        eval_examples = eval_examples.select(range(max_eval_samples))
    
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not config.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )

    # Custom data collator to handle None values in offset_mapping
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
    
    # Data collator - always use custom QA collator to handle None values
    data_collator = QADataCollator(tokenizer, pad_to_multiple_of=8 if config.fp16 else None)

    # Create post-processing and compute metrics functions using training_utils
    from training_utils import create_post_processing_function, create_compute_metrics_function
    
    # Create training args object for compatibility
    class TrainingArgs:
        def __init__(self, config):
            self.output_dir = config.output_dir
    
    training_args = TrainingArgs(config)
    
    # Create post-processing function
    post_processing_function = create_post_processing_function(
        data_args, training_args, answer_column_name, logging.INFO
    )
    
    # Create compute metrics function
    class ModelArgs:
        def __init__(self, config):
            self.cache_dir = config.cache_dir
    
    model_args = ModelArgs(config)
    compute_metrics = create_compute_metrics_function(data_args, model_args)

    # Create evaluation data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=config.dataloader_num_workers,
    )

    # Get layer names and generate sensitivity configs
    test_bits = args.test_bits  # Bits to test for each layer
    
    if args.layer_config_path:
        # Use provided config file
        base_config = load_base_config(args.layer_config_path)
        # Extract layer names from config
        layer_names = set()
        if 'a_bits' in base_config:
            layer_names.update(base_config['a_bits'].keys())
        if 'w_bits' in base_config:
            layer_names.update(base_config['w_bits'].keys())
        layer_names = list(layer_names)
        logging.info(f"Using layer names from config file: {len(layer_names)} layers")
    else:
        # Extract layer names from model
        layer_names = get_model_layers(model)
        logging.info(f"Auto-detected layer names from model: {len(layer_names)} layers")
        print(f"Auto-detected quantizable layers: {layer_names}")
    
    layer_configs = generate_layer_sensitivity_configs(layer_names, test_bits)
    
    # Store all results
    all_results = {}
    best_config = None
    best_metrics = None
    best_score = 0.0
    
    # Track results per layer for incremental chart creation
    layer_results = {}
    
    logging.info(f"Testing {len(layer_configs)} different layer configurations...")
    print(f"Testing {len(layer_configs)} different layer configurations...")
    
    for config_name, layer_config in layer_configs.items():
        logging.info(f"Testing configuration: {config_name}")
        print(f"Testing configuration: {config_name}")
        
        # Create DataParallel-compatible quant_config
        dataparallel_config = create_dataparallel_quant_config(layer_config)
        
        # Debug: Print quant_config for non-baseline configs
        if config_name != 'baseline':
            print(f"DEBUG: Config {config_name}")
            print(f"DEBUG: quant_config keys: {list(dataparallel_config.keys())}")
            if 'a_bit' in dataparallel_config:
                non_16_bits = {k: v for k, v in dataparallel_config['a_bit'].items() if v != 16}
                if non_16_bits:
                    print(f"DEBUG: Non-16-bit a_bit entries: {non_16_bits}")
            if 'w_bit' in dataparallel_config:
                non_16_bits = {k: v for k, v in dataparallel_config['w_bit'].items() if v != 16}
                if non_16_bits:
                    print(f"DEBUG: Non-16-bit w_bit entries: {non_16_bits}")
        
        # Evaluate model with current configuration
        eval_metrics = evaluate_model(
            model, eval_loader, eval_examples, eval_dataset,
            post_processing_function, compute_metrics,
            logger, config_name, dataparallel_config
        )
        
        # Store results
        all_results[config_name] = eval_metrics
        
        # Log metrics
        for key, value in eval_metrics.items():
            logger.add_scalar(f'eval/{config_name}/{key}', value, 0)
            logging.info(f"Config {config_name}: {key} = {value:.4f}")
        
        # Determine best configuration based on F1 score (or another metric)
        current_score = eval_metrics.get('f1', 0.0)
        if current_score > best_score:
            best_score = current_score
            best_config = config_name
            best_metrics = eval_metrics
            logging.info(f"New best configuration: {config_name} with F1 = {best_score:.4f}")
        
        # Save metrics for this configuration
        eval_metrics["config_name"] = config_name
        eval_metrics["eval_samples"] = len(eval_dataset)
        save_metrics_to_json(eval_metrics, config.save, f"eval_{config_name}")
        
        # Track results per layer for incremental chart creation
        if config_name != 'baseline':
            # Parse config name to extract layer and bit
            if config_name.endswith('bit'):
                parts = config_name.rsplit('_', 1)
                if len(parts) == 2:
                    layer_name = parts[0]
                    bit_str = parts[1]  # e.g., "4bit"
                    bit = int(bit_str[:-3])  # Remove "bit" suffix
                    
                    if layer_name not in layer_results:
                        layer_results[layer_name] = {}
                    layer_results[layer_name][bit] = eval_metrics.get('f1', 0.0)
                    
                    # Check if we've completed all bit combinations for this layer
                    if len(layer_results[layer_name]) == len(test_bits):
                        # Create and save chart for this layer
                        create_layer_chart(layer_name, layer_results[layer_name], test_bits, config.save)
                        logging.info(f"Created chart for layer: {layer_name}")
                        print(f"Created chart for layer: {layer_name}")
    
    # Create heatmaps
    print("Creating heatmaps...")
    create_heatmaps(all_results, config.save, test_bits)
    
    # Log final results
    logging.info(f"Best configuration: {best_config}")
    logging.info(f"Best F1 score: {best_score:.4f}")
    logging.info(f"Best metrics: {best_metrics}")
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best configuration: {best_config}")
    print(f"Best F1 score: {best_score:.4f}")
    print(f"Best metrics: {best_metrics}")
    
    # Save final results
    final_results = {
        "best_config": best_config,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "all_configs": list(layer_configs.keys()),
        "all_results": all_results
    }
    save_metrics_to_json(final_results, config.save, "final_results")




def evaluate_model(model, eval_loader, eval_examples, eval_dataset, post_processing_function, compute_metrics, logger, config_name, quant_config):
    """Evaluate model with current layer configuration"""
    model.eval()
    device = next(model.parameters()).device
    ignore_keys = ["loss"]  # Keys to ignore when extracting logits
    
    all_predictions = []
    #print(f"Quant config: {quant_config}")
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating config: {config_name}"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Remove offset_mapping from batch before passing to model (it's only needed for post-processing)
            model_batch = {k: v for k, v in batch.items() if k != 'offset_mapping'}
            
            # Forward pass with current layer configuration
            
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
    
    # Concatenate all predictions (like custom_trainer.py)
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
    
    # Post-process predictions exactly like custom_trainer.py
    if post_processing_function is not None and compute_metrics is not None:
        eval_preds = post_processing_function(eval_examples, eval_dataset, predictions)
        metrics = compute_metrics(eval_preds)
    else:
        metrics = {}
    
    print(f"Config {config_name} - Eval Metrics: {metrics}")
    
    return metrics

if __name__ == '__main__':
    main()