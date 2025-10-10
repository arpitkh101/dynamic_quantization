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

# Quantization and LoRA arguments
parser.add_argument('--a_bits', nargs='+', type=int, default=[8, 4],
                    help='activation bit widths')
parser.add_argument('--w_bits', nargs='+', type=int, default=[8, 4],
                    help='weight bit widths')
parser.add_argument('--ranks', nargs='+', type=int, default=[8, 8],
                    help='LoRA ranks')
parser.add_argument('--alphas', nargs='+', type=float, default=[16.0, 16.0],
                    help='LoRA alphas')
parser.add_argument('--dropouts', nargs='+', type=float, default=[0.01, 0.01],
                    help='LoRA dropouts')
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

    gpt2_config.a_bits = config.a_bits
    gpt2_config.w_bits = config.w_bits
    gpt2_config.ranks = config.ranks
    gpt2_config.alphas = config.alphas
    gpt2_config.dropouts = config.dropouts
    gpt2_config.kv_bits = config.kv_bits

    model = load_model(gpt2_config)
    distill_loss_fn = QADistillationLoss(temperature=2.0)
    
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
    
    # if config.lr_schedule == 'linear':
    #     def lr_lambda(current_step):
    #         if current_step < warmup_steps:
    #             return float(current_step) / float(max(1, warmup_steps))
    #         return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
    #     lr_policy = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # elif config.lr_schedule == 'cosine':
    #     lr_policy = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.learning_rate_min)
    # else:
    #     logging.info("Wrong Learning Rate Schedule Type.")
    #     sys.exit()

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
        train_loss = train_by_steps(
            train_loader, model, optimizer, lr_policy, logger, 
            a_bits=config.a_bits,
            w_bits=config.w_bits,
            bit_schedule=config.bit_schedule,
            loss_scale=config.loss_scale, 
            distill_weight=config.distill_weight, 
            cascad=config.cascad,
            max_steps=total_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            eval_loader=eval_loader if config.do_eval else None,
            eval_examples=eval_examples if config.do_eval else None,
            eval_dataset=eval_dataset if config.do_eval else None,
            post_processing_function=post_processing_function if config.do_eval else None,
            compute_metrics=compute_metrics if config.do_eval else None,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            distill_loss_fn=distill_loss_fn

        )
        
        # Save training metrics
        train_metrics = {
            "train_loss": train_loss,
            "train_samples": len(train_dataset),
            "total_steps": total_steps
        }
        save_metrics_to_json(train_metrics, config.save, "train")
        
    else:
        # Training by epochs (original logic)
        start_epoch = 0
        checkpoint = config.resume_from_checkpoint or last_checkpoint
        if checkpoint:
            start_epoch, _, _ = load_training_state(model, optimizer, lr_policy, checkpoint)
        
        for epoch in range(start_epoch, config.num_train_epochs):
            logging.info("Epoch %d/%d", epoch + 1, config.num_train_epochs)
            logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

            if config.do_train:
                # Training
                train_loss = train_epoch(
                    train_loader, model, optimizer, lr_policy, logger, epoch, 
                    a_bits=config.a_bits,
                    w_bits=config.w_bits,
                    bit_schedule=config.bit_schedule,
                    loss_scale=config.loss_scale, 
                    distill_weight=config.distill_weight, 
                    cascad=config.cascad,
                    distill_loss_fn=distill_loss_fn
                )
                
                logging.info("Epoch %d: Train Loss = %.4f", epoch, train_loss)
                
                # Clear GPU cache and step learning rate scheduler (like original train.py)
                torch.cuda.empty_cache()
                #lr_policy.step()

            # Evaluation
            if config.do_eval and (epoch + 1) % config.eval_epoch == 0:
                eval_metrics = evaluate_model(
                    model, eval_loader, eval_examples, eval_dataset, 
                    post_processing_function, compute_metrics, 
                    config.a_bits, config.w_bits, logger, epoch
                )
                
                # Log metrics
                for key, value in eval_metrics.items():
                    logger.add_scalar(f'eval/{key}', value, epoch)
                    logging.info("Epoch %d: %s = %.4f", epoch, key, value)
                
                # Save best model
                if config.load_best_model_at_end:
                    current_metric = eval_metrics.get(config.metric_for_best_model, 0.0)
                    if current_metric > best_metric:
                        best_metric = current_metric
                        save_model(model, os.path.join(config.save, 'best_model.pt'))
                        logging.info("New best model saved with %s = %.4f", config.metric_for_best_model, best_metric)

            # Save checkpoint
            save_model(model, os.path.join(config.save, f'checkpoint-epoch-{epoch}.pt'))

    # Final evaluation
    if config.do_eval:
        final_metrics = evaluate_model(
            model, eval_loader, eval_examples, eval_dataset,
            post_processing_function, compute_metrics,
            config.a_bits, config.w_bits, logger, config.num_train_epochs
        )
        logging.info('Final Eval - Metrics: %s', str(final_metrics))
        
        # Save evaluation metrics
        final_metrics["eval_samples"] = len(eval_dataset)
        save_metrics_to_json(final_metrics, config.save, "eval")

    # Save final model and tokenizer
    save_model(model, os.path.join(config.save, 'final_model.pt'))
    save_tokenizer(tokenizer, config.save)


def train_by_steps(train_loader, model, optimizer, lr_policy, logger, a_bits, w_bits, bit_schedule, loss_scale, distill_weight, cascad, max_steps, eval_steps, save_steps, eval_loader=None, eval_examples=None, eval_dataset=None, post_processing_function=None, compute_metrics=None, start_epoch=0, start_global_step=0, distill_loss_fn=None):
    """Train by steps with switch precision"""
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
        
        optimizer.zero_grad()
        
        loss_value = [-1 for _ in a_bits]
        
        if bit_schedule == 'avg_loss':
            if distill_weight > 0:
                if cascad:
                    teacher_list = []
                    for i, (a_bit, w_bit) in enumerate(zip(a_bits[::-1], w_bits[::-1])):
                        outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        
                        if len(teacher_list) > 0:
                            for logit_teacher in teacher_list:
                                #loss += distill_weight * (nn.MSELoss()(outputs.start_logits, logit_teacher['start_logits']) + 
                                #                        nn.MSELoss()(outputs.end_logits, logit_teacher['end_logits']))/2
                                loss += distill_weight * distill_loss_fn(outputs.start_logits, outputs.end_logits, logit_teacher['start_logits'], logit_teacher['end_logits'])
                        
                        # Store teacher outputs for distillation
                        teacher_list.append({
                            'start_logits': outputs.start_logits.detach(),
                            'end_logits': outputs.end_logits.detach()
                        })
                        #print(loss)
                        loss = loss.mean()
                        #print(loss)
                        loss = loss * loss_scale[len(a_bits) - 1 - i]
                        loss.backward()
                        
                        loss_value[len(a_bits) - 1 - i] = loss.item()
                        
                        del outputs
                        del loss
                else:
                    # Teacher-student distillation
                    outputs = model(**batch, a_bit=a_bits[-1], w_bit=w_bits[-1])
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    loss = loss * loss_scale[-1]
                    loss.backward()
                    loss_value[-1] = loss.item()
                    
                    teacher_outputs = {
                        'start_logits': outputs.start_logits.detach(),
                        'end_logits': outputs.end_logits.detach()
                    }
                    del outputs
                    del loss
                    
                    for i, (a_bit, w_bit) in enumerate(zip(a_bits[:-1], w_bits[:-1])):
                        outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        # loss += distill_weight * (nn.MSELoss()(outputs.start_logits, teacher_outputs['start_logits']) + 
                        #                          nn.MSELoss()(outputs.end_logits, teacher_outputs['end_logits']))
                        loss += distill_weight * distill_loss_fn(outputs.start_logits, outputs.end_logits, teacher_outputs['start_logits'], teacher_outputs['end_logits'])
                        
                        loss = loss.mean()
                        loss = loss * loss_scale[len(a_bits) - 1 - i]
                        loss.backward()
                        
                        loss_value[len(a_bits) - 1 - i] = loss.item()
                        
                        del outputs
                        del loss
            else:
                # No distillation
                for i, (a_bit, w_bit) in enumerate(zip(a_bits, w_bits)):
                    outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    loss = loss * loss_scale[i]
                    loss.backward()
                    
                    loss_value[i] = loss.item()
                    
                    del outputs
                    del loss
            
            # Gradient clipping and optimizer step
            if config.grad_clip > 0:
                # Handle DataParallel case for gradient clipping
                if hasattr(model, 'module'):
                    nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            lr_policy.step()
            
        else:
            # High to low bit schedule: train with each bit width separately
            for i, (a_bit, w_bit) in enumerate(zip(sorted(a_bits, reverse=True), sorted(w_bits, reverse=True))):
                outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                loss = loss * loss_scale[a_bits.index(a_bit)]
                loss.backward()
                
                # Step optimizer after each bit width (like original train.py)
                # Handle DataParallel case for gradient clipping
                if hasattr(model, 'module'):
                    nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                lr_policy.step()
                optimizer.zero_grad()
                
                loss_value[a_bits.index(a_bit)] = loss.item()
        
            # Other bit schedules (low2high, sandwich, max_loss)
            #logging.warning("Bit schedule %s not fully implemented, using avg_loss", bit_schedule)
        
        # Log losses
        for i, (a_bit, w_bit) in enumerate(zip(a_bits, w_bits)):
            if loss_value[i] != -1:
                logger.add_scalar(f'loss/a_bits_{a_bit}_w_bits_{w_bit}', loss_value[i], global_step)
                total_loss += loss_value[i]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{sum([v for v in loss_value if v != -1]) / len([v for v in loss_value if v != -1]):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'step': f'{global_step}/{max_steps}'
        })
        
        global_step += 1
        
        # Evaluation
        if eval_loader is not None and eval_steps > 0 and global_step % eval_steps == 0:
            eval_metrics = evaluate_model(
                model, eval_loader, eval_examples, eval_dataset,
                post_processing_function, compute_metrics,
                a_bits, w_bits, logger, global_step
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


def train_epoch(train_loader, model, optimizer, lr_policy, logger, epoch, a_bits, w_bits, bit_schedule, loss_scale, distill_weight, cascad, distill_loss_fn):
    """Train for one epoch with switch precision"""
    model.train()
    
    total_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
    
    for step, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        loss_value = [-1 for _ in a_bits]
        
        if bit_schedule == 'avg_loss':
            if distill_weight > 0:
                if cascad:
                    teacher_list = []
                    for i, (a_bit, w_bit) in enumerate(zip(a_bits[::-1], w_bits[::-1])):
                        outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        loss = loss.mean()
                        
                        if len(teacher_list) > 0:
                            for logit_teacher in teacher_list:
                                # For QA models, distill both start and end logits
                                # loss += distill_weight * (nn.MSELoss()(outputs.start_logits, logit_teacher['start_logits']) + 
                                #                          nn.MSELoss()(outputs.end_logits, logit_teacher['end_logits']))
                                loss += distill_weight * distill_loss_fn(outputs.start_logits, outputs.end_logits, logit_teacher['start_logits'], logit_teacher['end_logits'])
                        
                        # Store teacher outputs for distillation
                        teacher_list.append({
                            'start_logits': outputs.start_logits.detach(),
                            'end_logits': outputs.end_logits.detach()
                        })
                        
                        loss = loss * loss_scale[len(a_bits) - 1 - i]
                        #print(loss)
                        loss.backward()
                        
                        loss_value[len(a_bits) - 1 - i] = loss.item()
                        
                        del outputs
                        del loss
                else:
                    # Teacher-student distillation
                    outputs = model(**batch, a_bit=a_bits[-1], w_bit=w_bits[-1])
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    loss = loss * loss_scale[-1]
                    loss.backward()
                    loss_value[-1] = loss.item()
                    
                    teacher_outputs = {
                        'start_logits': outputs.start_logits.detach(),
                        'end_logits': outputs.end_logits.detach()
                    }
                    del outputs
                    del loss
                    
                    for i, (a_bit, w_bit) in enumerate(zip(a_bits[:-1], w_bits[:-1])):
                        outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                        # loss += distill_weight * (nn.MSELoss()(outputs.start_logits, teacher_outputs['start_logits']) + 
                        #                          nn.MSELoss()(outputs.end_logits, teacher_outputs['end_logits']))
                        loss += distill_weight * distill_loss_fn(outputs.start_logits, outputs.end_logits, teacher_outputs['start_logits'], teacher_outputs['end_logits'])
                        
                        loss = loss * loss_scale[len(a_bits) - 1 - i]
                        loss.backward()
                        
                        loss_value[len(a_bits) - 1 - i] = loss.item()
                        
                        del outputs
                        del loss
            
            else:
                # No distillation
                for i, (a_bit, w_bit) in enumerate(zip(a_bits, w_bits)):
                    outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    loss = loss * loss_scale[i]
                    loss.backward()
                    
                    loss_value[i] = loss.item()
                    
                    del outputs
                    del loss
            
            # Gradient clipping and optimizer step
            if config.grad_clip > 0:
                # Handle DataParallel case for gradient clipping
                if hasattr(model, 'module'):
                    nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            
            optimizer.step()
            lr_policy.step()
            
        else:
            # High to low bit schedule: train with each bit width separately
            for i, (a_bit, w_bit) in enumerate(zip(sorted(a_bits, reverse=True), sorted(w_bits, reverse=True))):
                outputs = model(**batch, a_bit=a_bit, w_bit=w_bit)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                loss = loss * loss_scale[a_bits.index(a_bit)]
                loss.backward()
                
                # Step optimizer after each bit width (like original train.py)
                # Handle DataParallel case for gradient clipping
                if hasattr(model, 'module'):
                    nn.utils.clip_grad_norm_(model.module.parameters(), config.grad_clip)
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                lr_policy.step()
                
                loss_value[a_bits.index(a_bit)] = loss.item()
        
            # Fallback to avg_loss implementation
        
        # Log losses
        for i, (a_bit, w_bit) in enumerate(zip(a_bits, w_bits)):
            if loss_value[i] != -1:
                logger.add_scalar(f'loss/a_bits_{a_bit}_w_bits_{w_bit}', loss_value[i], epoch * num_batches + step)
                total_loss += loss_value[i]
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{sum([v for v in loss_value if v != -1]) / len([v for v in loss_value if v != -1]):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    return total_loss / num_batches


def evaluate_model(model, eval_loader, eval_examples, eval_dataset, post_processing_function, compute_metrics, a_bits, w_bits, logger, epoch):
    """Evaluate model with switch precision at each bit-width"""
    model.eval()
    device = next(model.parameters()).device
    ignore_keys = ["loss"]  # Keys to ignore when extracting logits
    
    all_metrics = {}
    
    # Evaluate at each bit-width combination
    for a_bit, w_bit in zip(a_bits, w_bits):
        #logger.info(f"Evaluating at a_bit={a_bit}, w_bit={w_bit}")
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Evaluating a_bit={a_bit}, w_bit={w_bit}"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Remove offset_mapping from batch before passing to model (it's only needed for post-processing)
                model_batch = {k: v for k, v in batch.items() if k != 'offset_mapping'}
                
                # Forward pass with current bit width
                outputs = model(**model_batch, a_bit=a_bit, w_bit=w_bit)
                
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
        bit_key = f"a_bit_{a_bit}_w_bit_{w_bit}"
        for metric_name, metric_value in metrics.items():
            all_metrics[f"{bit_key}_{metric_name}"] = metric_value
        
        #logger.info(f"Epoch {epoch} - Eval Metrics at {bit_key}: {metrics}")
        print(f"Epoch {epoch} - Eval Metrics at {bit_key}: {metrics}")
    
    # Also compute average metrics across all bit-widths
    metric_names = set()
    for key in all_metrics.keys():
        if key.startswith("a_bit_"):
            metric_name = key.split("_", 4)[-1]  # Get the metric name after bit info
            metric_names.add(metric_name)
    
    for metric_name in metric_names:
        values = []
        for a_bit, w_bit in zip(a_bits, w_bits):
            key = f"a_bit_{a_bit}_w_bit_{w_bit}_{metric_name}"
            if key in all_metrics:
                values.append(all_metrics[key])
        if values:  # Only calculate average if we have values
            all_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
    
    #logger.info(f"Epoch {epoch} - Average Eval Metrics: {dict((k, v) for k, v in all_metrics.items() if k.startswith('avg_'))}")
    print(f"Epoch {epoch} - Average Eval Metrics: {dict((k, v) for k, v in all_metrics.items() if k.startswith('avg_'))}")
    
    return all_metrics

if __name__ == '__main__':
    main()