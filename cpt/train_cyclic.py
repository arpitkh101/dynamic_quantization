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

# Import our custom components from local models package
from models import GPT2Config, QuantizeLoraLinear
from utils import postprocess_qa_predictions
from training_utils import create_data_preprocessing_functions
import copy
from typing import List
# Removed QADistillationLoss - not using distillation
import argparse
from utils import (
    get_lora_params, count_trainable_parameters, freeze_model_for_lora_training,
    save_model, save_training_state, load_training_state, 
    save_metrics_to_json, save_tokenizer, load_model
)

parser = argparse.ArgumentParser(description='GPT-2 QA Training with Cyclic Precision')
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

# Training arguments
parser.add_argument('--do_train', action='store_true',
                    help='whether to run training')
parser.add_argument('--do_eval', action='store_true',
                    help='whether to run evaluation')
parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                    help='batch size per device for training')
parser.add_argument('--per_device_eval_batch_size', type=int, default=16,
                    help='batch size per device for evaluation')
parser.add_argument('--learning_rate', type=float, default=3e-3,
                    help='learning rate')
parser.add_argument('--num_train_epochs', type=int, default=3,
                    help='number of training epochs')
parser.add_argument('--max_steps', type=int, default=None,
                    help='maximum number of training steps')
parser.add_argument('--max_seq_length', type=int, default=786,
                    help='maximum sequence length')
parser.add_argument('--max_train_samples', type=int, default=None,
                    help='maximum number of training samples')
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

# Cyclic precision training arguments
parser.add_argument('--is_cyclic_precision', action='store_true',
                    help='enable cyclic precision training')
parser.add_argument('--cyclic_a_bits_schedule', default=None, type=int, nargs='*',
                    help='cyclic schedule for activation precision [min, max]')
parser.add_argument('--cyclic_w_bits_schedule', default=None, type=int, nargs='*',
                    help='cyclic schedule for weight precision [min, max]')
parser.add_argument('--num_cyclic_period', default=32, type=int,
                    help='number of cyclic periods for precision')

# Quantization and LoRA arguments (simplified for cyclic precision)
parser.add_argument('--a_bits', type=int, default=8,
                    help='default activation bit width (used when not cycling)')
parser.add_argument('--w_bits', type=int, default=8,
                    help='default weight bit width (used when not cycling)')
parser.add_argument('--ranks', type=int, default=16,
                    help='LoRA rank (single value for single adapter)')
parser.add_argument('--alphas', type=float, default=32.0,
                    help='LoRA alpha (single value for single adapter)')
parser.add_argument('--dropouts', type=float, default=0.01,
                    help='LoRA dropout (single value for single adapter)')
parser.add_argument('--kv_bits', type=int, default=16,
                    help='key-value quantization bits')

args = parser.parse_args()

def main():
    # Update config from arguments
    from config_new import update_config_from_args
    update_config_from_args(args)
    
    # Setup logging and output directory
    config.save = '{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    config.output_dir = config.save
    
    # Add cyclic precision specific save path naming
    if config.is_cyclic_precision:
        config.save = f"{config.save}_cyclic_a_{config.cyclic_a_bits_schedule[0]}_{config.cyclic_a_bits_schedule[1]}_w_{config.cyclic_w_bits_schedule[0]}_{config.cyclic_w_bits_schedule[1]}_period_{config.num_cyclic_period}"
        config.output_dir = config.save
    
    last_checkpoint = None
    if os.path.isdir(config.output_dir) and config.do_train and not config.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint is None and len(os.listdir(config.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({config.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and not hasattr(config, 'resume_from_checkpoint'):
            logging.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `output_dir` or add `--overwrite_output_dir` to your run command."
            )
    
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

    gpt2_config = GPT2Config.from_pretrained(
        config.config_name if config.config_name else config.model_name_or_path,
        cache_dir=config.cache_dir,
        revision=config.model_revision,
        token=config.token,
        trust_remote_code=config.trust_remote_code,
    )

    # Set quantization config for cyclic precision
    if config.is_cyclic_precision:
        # Use the maximum values from the cyclic schedule for model initialization
        gpt2_config.a_bits = config.cyclic_a_bits_schedule[1]
        gpt2_config.w_bits = config.cyclic_w_bits_schedule[1]
    else:
        gpt2_config.a_bits = config.a_bits
        gpt2_config.w_bits = config.w_bits
    
    gpt2_config.ranks = config.ranks
    gpt2_config.alphas = config.alphas
    gpt2_config.dropouts = config.dropouts
    gpt2_config.kv_bits = config.kv_bits

    model = load_model(gpt2_config)
    
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
    
    freeze_model_for_lora_training(model, config)

    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "LayerNorm.weight" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    betas = getattr(config, "betas", (0.9, 0.999))

    if config.opt == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=betas
        )
    elif config.opt == "Adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=betas
        )
    else:
        logging.info("Wrong Optimizer Type.")   
        sys.exit()

    # Setup learning rate scheduler
    total_steps = config.num_train_epochs * 1000  # Will be updated with actual dataset size
    warmup_steps = config.warmup_steps
    
    if config.lr_schedule == 'linear':
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

        lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 2️⃣ Cosine schedule (with warmup)
    elif config.lr_schedule == 'cosine':
        # Linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        # Cosine decay
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=getattr(config, "learning_rate_min", 0.0)
        )
        lr_policy = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
    else:
        logging.info("Wrong Learning Rate Schedule Type.")
        sys.exit()

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

    # Get column names
    if config.do_train:
        column_names = raw_datasets["train"].column_names
    elif config.do_eval:
        column_names = raw_datasets["validation"].column_names
    else:
        column_names = raw_datasets["test"].column_names if "test" in raw_datasets else raw_datasets["validation"].column_names
    
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

    # Prepare training dataset
    if config.do_train:
        train_dataset = raw_datasets["train"]
        if config.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), config.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=config.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not config.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    # Prepare evaluation dataset
    if config.do_eval:
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

    # Create data loaders
    if config.do_train:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=config.dataloader_num_workers,
        )
        config.niters_per_epoch = len(train_loader)

    if config.do_eval:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=config.dataloader_num_workers,
        )

    # Training loop
    global_step = 0
    best_metric = 0.0
    
    # Determine training strategy (epochs vs max_steps)
    if config.max_steps is not None:
        # Training by steps
        total_steps = config.max_steps
        logging.info("Training for %d steps", total_steps)
        
        # Check for checkpoint resume
        start_epoch = 0
        start_global_step = 0
        checkpoint = config.resume_from_checkpoint or last_checkpoint
        if checkpoint:
            start_epoch, start_global_step, _ = load_training_state(model, optimizer, lr_policy, checkpoint)
        
        # Training loop by steps
        train_loss = train_by_steps_cyclic(
            train_loader, model, optimizer, lr_policy, logger, 
            config=config,
            max_steps=total_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            eval_loader=eval_loader if config.do_eval else None,
            eval_examples=eval_examples if config.do_eval else None,
            eval_dataset=eval_dataset if config.do_eval else None,
            post_processing_function=post_processing_function if config.do_eval else None,
            compute_metrics=compute_metrics if config.do_eval else None,
            start_epoch=start_epoch,
            start_global_step=start_global_step
        )
        
        # Save training metrics
        train_metrics = {
            "train_loss": train_loss,
            "train_samples": len(train_dataset),
            "total_steps": total_steps
        }
        save_metrics_to_json(train_metrics, config.save, "train")
        
    # Only step-based training is supported

    # Final evaluation
    if config.do_eval:
        final_metrics = evaluate_model_cyclic(
            model, eval_loader, eval_examples, eval_dataset,
            post_processing_function, compute_metrics,
            config, logger, config.num_train_epochs
        )
        logging.info('Final Eval - Metrics: %s', str(final_metrics))
        
        # Save evaluation metrics
        final_metrics["eval_samples"] = len(eval_dataset)
        save_metrics_to_json(final_metrics, config.save, "eval")

    # Save final model and tokenizer
    save_model(model, os.path.join(config.save, 'final_model.pt'))
    save_tokenizer(tokenizer, config.save)


def cyclic_adjust_precision(config, _iter, cyclic_period):
    """Adjust precision based on cyclic schedule"""
    if config.is_cyclic_precision:
        assert len(config.cyclic_a_bits_schedule) == 2
        assert len(config.cyclic_w_bits_schedule) == 2

        a_bit_min = config.cyclic_a_bits_schedule[0]
        a_bit_max = config.cyclic_a_bits_schedule[1]

        w_bit_min = config.cyclic_w_bits_schedule[0]
        w_bit_max = config.cyclic_w_bits_schedule[1]

        # Use cosine scheduling for smooth transitions
        config.current_a_bits = int(np.rint(a_bit_min +
                                0.5 * (a_bit_max - a_bit_min) *
                                (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi))))
        config.current_w_bits = int(np.rint(w_bit_min +
                                0.5 * (w_bit_max - w_bit_min) *
                                (1 + np.cos(np.pi * ((_iter % cyclic_period) / cyclic_period) + np.pi))))
        
        if _iter % 10 == 0:
            logging.info('Iter [{}] a_bits = {} w_bits = {} cyclic precision'.format(_iter, config.current_a_bits, config.current_w_bits))
    else:
        # Use default precision values
        config.current_a_bits = config.a_bits
        config.current_w_bits = config.w_bits


def train_by_steps_cyclic(train_loader, model, optimizer, lr_policy, logger, config, max_steps, eval_steps, save_steps, eval_loader=None, eval_examples=None, eval_dataset=None, post_processing_function=None, compute_metrics=None, start_epoch=0, start_global_step=0):
    """Train by steps with cyclic precision"""
    model.train()
    
    global_step = start_global_step
    best_metric = 0.0
    total_loss = 0.0
    
    # Create infinite iterator for training data
    train_iter = iter(train_loader)
    
    pbar = tqdm(range(max_steps), desc="Training")
    
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            # Restart iterator when exhausted
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Move batch to device
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Calculate cyclic period and adjust precision
        # Use max_steps instead of epochs for step-based training
        #cyclic_period = int(max_steps / config.num_cyclic_period)
        #print(f"len(train_loader): {len(train_loader)}")
        cyclic_period = int(len(train_loader) / config.num_cyclic_period)
        _iter = global_step
        cyclic_adjust_precision(config, _iter, cyclic_period)
        
        optimizer.zero_grad()
        
        # Forward pass with current precision
        outputs = model(**batch, a_bit=config.current_a_bits, w_bit=config.current_w_bits)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Ensure loss is a scalar
        if loss.dim() > 0:
            loss = loss.mean()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            if hasattr(model, 'module'):
                nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        lr_policy.step()
        
        # Log loss
        logger.add_scalar(f'loss/a_bits_{config.current_a_bits}_w_bits_{config.current_w_bits}', loss.item(), global_step)
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'a_bits': config.current_a_bits,
            'w_bits': config.current_w_bits,
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'step': f'{global_step}/{max_steps}'
        })
        
        global_step += 1
        
        # Evaluation
        if eval_loader is not None and eval_steps > 0 and global_step % eval_steps == 0:
            eval_metrics = evaluate_model_cyclic(
                model, eval_loader, eval_examples, eval_dataset,
                post_processing_function, compute_metrics,
                config, logger, global_step
            )
            
            # Log metrics
            for key, value in eval_metrics.items():
                logger.add_scalar(f'eval/{key}', value, global_step)
                logging.info("Step %d: %s = %.4f", global_step, key, value)
            
            # Save best model
            if config.load_best_model_at_end:
                current_metric = eval_metrics.get(config.metric_for_best_model, 0.0)
                if current_metric > best_metric:
                    best_metric = current_metric
                    save_model(model, os.path.join(config.save, 'best_model.pt'))
                    logging.info("New best model saved with %s = %.4f", config.metric_for_best_model, best_metric)
            
            model.train()  # Return to training mode
        
        # Save checkpoint
        if save_steps > 0 and global_step % save_steps == 0:
            metrics = {'train_loss': total_loss / (global_step + 1)}
            save_training_state(model, optimizer, lr_policy, 0, global_step, metrics, config.save)
    
    return total_loss / max_steps


# Removed train_epoch_cyclic function - only step-based training is supported


def evaluate_model_cyclic(model, eval_loader, eval_examples, eval_dataset, post_processing_function, compute_metrics, config, logger, epoch):
    """Evaluate model with cyclic precision at all bit combinations"""
    model.eval()
    device = next(model.parameters()).device
    ignore_keys = ["loss"]  # Keys to ignore when extracting logits
    
    all_metrics = {}
    
    # Determine bit combinations to evaluate
    if config.is_cyclic_precision:
        # Evaluate at all bit combinations in the cyclic schedule
        a_bits_range = list(range(config.cyclic_a_bits_schedule[0], config.cyclic_a_bits_schedule[1] + 1))
        w_bits_range = list(range(config.cyclic_w_bits_schedule[0], config.cyclic_w_bits_schedule[1] + 1))
    else:
        # Evaluate at single precision
        a_bits_range = [config.a_bits]
        w_bits_range = [config.w_bits]
    
    # Evaluate at each bit combination
    for eval_a_bits, eval_w_bits in zip(a_bits_range, w_bits_range):
        print(f"Evaluating at a_bit={eval_a_bits}, w_bit={eval_w_bits}...")
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluating a_bit={eval_a_bits}, w_bit={eval_w_bits}"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Remove offset_mapping from batch before passing to model (it's only needed for post-processing)
                model_batch = {k: v for k, v in batch.items() if k != 'offset_mapping'}
                
                # Forward pass with current bit width
                outputs = model(**model_batch, a_bit=eval_a_bits, w_bit=eval_w_bits)
                
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
        
        # Store metrics with bit-width prefix
        bit_key = f"a_bit_{eval_a_bits}_w_bit_{eval_w_bits}"
        for metric_name, metric_value in metrics.items():
            all_metrics[f"{bit_key}_{metric_name}"] = metric_value
        
        print(f"Epoch {epoch} - Eval Metrics at {bit_key}: {metrics}")

    # Also compute average metrics across all bit-widths
    metric_names = set()
    for key in all_metrics.keys():
        if key.startswith("a_bit_"):
            metric_name = key.split("_", 4)[-1]  # Get the metric name after bit info
            metric_names.add(metric_name)
    
    for metric_name in metric_names:
        values = []
        for eval_a_bits in a_bits_range:
            for eval_w_bits in w_bits_range:
                key = f"a_bit_{eval_a_bits}_w_bit_{eval_w_bits}_{metric_name}"
                if key in all_metrics:
                    values.append(all_metrics[key])
        if values:  # Only calculate average if we have values
            all_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
    
    print(f"Epoch {epoch} - Average Eval Metrics: {dict((k, v) for k, v in all_metrics.items() if k.startswith('avg_'))}")
    
    return all_metrics

if __name__ == '__main__':
    main()
