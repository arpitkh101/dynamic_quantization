# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys
import time
import numpy as np
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


C = edict()
config = C
cfg = C

C.seed = 42
C.repo_name = 'switch_precision_qa'

"""Data Dir and Weight Dir"""
C.dataset_path = None  # Will be set from command line or run_qa.py
C.dataset = 'squad'  # Default QA dataset
C.dataset_name = 'squad'  # Dataset name for HuggingFace datasets
C.dataset_config_name = None  # Dataset config (e.g., 'plain_text')

"""QA Dataset Config"""
C.max_seq_length = 384  
C.doc_stride = 128
C.n_best_size = 20
C.max_answer_length = 30
C.version_2_with_negative = False
C.null_score_diff_threshold = 0.0

"""Model Config - GPT-2 specific"""
C.model_name_or_path = "gpt2"
C.config_name = None
C.tokenizer_name = None
C.cache_dir = None
C.model_revision = "main"
C.token = None
C.trust_remote_code = False

"""LoRA and Quantization Config"""
C.a_bits = [4, 8]  # Activation bit widths for switch precision
C.w_bits = [4, 8]  # Weight bit widths for switch precision
C.ranks = [8, 8]   # LoRA ranks for each bit width
C.alphas = [16.0, 16.0]  # LoRA alphas for each bit width
C.dropouts = [0.01, 0.01]  # LoRA dropouts for each bit width
C.kv_bits = 16  # Key-value quantization bits

"""Train Config"""
C.opt = 'AdamW'  # Changed from SGD to AdamW for transformer training
C.momentum = 0.9
C.weight_decay = 0.01  # Typical for transformer training
C.betas = (0.9, 0.999)
C.num_workers = 4

"""Switch Precision Training Config"""
C.grad_clip = 1.0  # Lower for transformer training
C.pretrain = False
C.bit_schedule = 'avg_loss' 
C.dws_chwise_quant = True

# Switch precision specific settings
C.num_bits_list = C.a_bits  # Use a_bits as the bit width list
C.loss_scale = [1.0, 1.0]  # Equal weight for each bit width
C.distill_weight = 1.0  # Knowledge distillation weight
C.cascad = True  # Use cascaded distillation
C.update_bn_freq = None  # Not applicable for transformers

C.num_bits_list_schedule = None
C.schedule_freq = 20

"""Training Hyperparameters - Defaults, can be overridden by command line"""
C.batch_size = 8  # Smaller batch size for transformer training
C.per_device_train_batch_size = 8
C.per_device_eval_batch_size = 16
C.gradient_accumulation_steps = 1
C.num_train_epochs = 3
C.max_steps = None  # Can override num_train_epochs
C.learning_rate = 3e-3  # Updated to match your command
C.warmup_steps = 500
C.eval_steps = 1000
C.save_steps = 1000
C.logging_steps = 50

"""Learning Rate Schedule"""
C.lr_schedule = 'cosine'  # Updated to match your command
C.lr_scheduler_type = 'cosine'  # HuggingFace style
C.lr = C.learning_rate
C.lr_decay = 0.97
C.milestones = [80, 120, 160]
C.gamma = 0.1
C.learning_rate_min = 1e-6

"""Evaluation and Logging"""
C.eval_epoch = 1  # Evaluate every epoch
C.eval_strategy = 'steps'  # HuggingFace style
C.eval_only = False
C.update_bn = False  # Not applicable for transformers
C.show_distrib = False
C.finetune_bn = False  # Not applicable for transformers
C.ft_bn_epoch = 10
C.ft_bn_lr = 1e-3
C.ft_bn_momentum = 0.9

"""Logging and Saving Strategy"""
C.logging_strategy = 'steps'  # HuggingFace style
C.save_strategy = 'steps'  # HuggingFace style

"""Output and Checkpointing"""
C.save = "gpt2_qa_switch_precision"
C.output_dir = "./outputs"
C.overwrite_output_dir = True
C.load_path = None  # Will be set if loading from checkpoint
C.resume_from_checkpoint = None

"""Data Processing"""
C.max_train_samples = None
C.max_eval_samples = None
C.max_predict_samples = None
C.preprocessing_num_workers = 4
C.overwrite_cache = False
C.pad_to_max_length = True

"""Training Control"""
C.do_train = True
C.do_eval = True
C.do_predict = False
C.prediction_loss_only = False
C.dataloader_drop_last = False
C.dataloader_num_workers = 0
C.past_index = -1
C.run_name = None
C.disable_tqdm = False
C.remove_unused_columns = True
C.label_names = None
C.load_best_model_at_end = True
C.metric_for_best_model = "f1"
C.greater_is_better = True
C.ignore_data_skip = False
C.should_save = True
C.save_total_limit = None
C.seed = 42
C.data_seed = None
C.bf16 = False
C.fp16 = False
C.fp16_opt_level = "O1"
C.local_rank = -1
C.ddp_find_unused_parameters = None
C.dataloader_pin_memory = True
C.skip_memory_metrics = True
C.use_legacy_prediction_loop = False
C.prediction_loss_only = False
C.include_inputs_for_metrics = False


def update_config_from_args(args):
    """Update config from command line arguments or run_qa.py parameters"""
    # Dataset and model parameters
    if hasattr(args, 'dataset_path') and args.dataset_path:
        C.dataset_path = args.dataset_path
    if hasattr(args, 'dataset_name') and args.dataset_name:
        C.dataset_name = args.dataset_name
    if hasattr(args, 'dataset_config_name') and args.dataset_config_name:
        C.dataset_config_name = args.dataset_config_name
    if hasattr(args, 'model_name_or_path') and args.model_name_or_path:
        C.model_name_or_path = args.model_name_or_path
    if hasattr(args, 'config_name') and args.config_name:
        C.config_name = args.config_name
    if hasattr(args, 'tokenizer_name') and args.tokenizer_name:
        C.tokenizer_name = args.tokenizer_name
    if hasattr(args, 'cache_dir') and args.cache_dir:
        C.cache_dir = args.cache_dir
    if hasattr(args, 'model_revision') and args.model_revision:
        C.model_revision = args.model_revision
    if hasattr(args, 'token') and args.token:
        C.token = args.token
    if hasattr(args, 'trust_remote_code') and args.trust_remote_code:
        C.trust_remote_code = args.trust_remote_code
    if hasattr(args, 'output_dir') and args.output_dir:
        C.output_dir = args.output_dir
        C.save = args.output_dir
    
    # Training parameters
    if hasattr(args, 'num_train_epochs') and args.num_train_epochs:
        C.num_train_epochs = args.num_train_epochs
    if hasattr(args, 'max_steps') and args.max_steps:
        C.max_steps = args.max_steps
    if hasattr(args, 'per_device_train_batch_size') and args.per_device_train_batch_size:
        C.batch_size = args.per_device_train_batch_size
        C.per_device_train_batch_size = args.per_device_train_batch_size
    if hasattr(args, 'per_device_eval_batch_size') and args.per_device_eval_batch_size:
        C.per_device_eval_batch_size = args.per_device_eval_batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        C.learning_rate = args.learning_rate
        C.lr = args.learning_rate
    
    # Learning rate scheduler
    if hasattr(args, 'lr_scheduler_type') and args.lr_scheduler_type:
        C.lr_scheduler_type = args.lr_scheduler_type
        # Map HuggingFace scheduler types to our internal types
        if args.lr_scheduler_type == 'cosine':
            C.lr_schedule = 'cosine'
        elif args.lr_scheduler_type == 'linear':
            C.lr_schedule = 'linear'
        elif args.lr_scheduler_type == 'constant':
            C.lr_schedule = 'constant'
    
    # Evaluation and logging
    if hasattr(args, 'eval_strategy') and args.eval_strategy:
        C.eval_strategy = args.eval_strategy
    if hasattr(args, 'eval_steps') and args.eval_steps:
        C.eval_steps = args.eval_steps
    if hasattr(args, 'logging_strategy') and args.logging_strategy:
        C.logging_strategy = args.logging_strategy
    if hasattr(args, 'logging_steps') and args.logging_steps:
        C.logging_steps = args.logging_steps
    if hasattr(args, 'save_strategy') and args.save_strategy:
        C.save_strategy = args.save_strategy
    if hasattr(args, 'save_steps') and args.save_steps:
        C.save_steps = args.save_steps
    
    # Dataset parameters
    if hasattr(args, 'max_seq_length') and args.max_seq_length:
        C.max_seq_length = args.max_seq_length
    if hasattr(args, 'max_train_samples') and args.max_train_samples:
        C.max_train_samples = args.max_train_samples
    if hasattr(args, 'max_eval_samples') and args.max_eval_samples:
        C.max_eval_samples = args.max_eval_samples
    if hasattr(args, 'overwrite_output_dir') and args.overwrite_output_dir:
        C.overwrite_output_dir = args.overwrite_output_dir
    if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
        C.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Quantization and LoRA parameters
    if hasattr(args, 'a_bits') and args.a_bits:
        C.a_bits = args.a_bits
        C.num_bits_list = args.a_bits
        C.loss_scale = [1.0 / len(args.a_bits)] * len(args.a_bits)
    if hasattr(args, 'w_bits') and args.w_bits:
        C.w_bits = args.w_bits
    if hasattr(args, 'ranks') and args.ranks:
        C.ranks = args.ranks
    if hasattr(args, 'alphas') and args.alphas:
        C.alphas = args.alphas
    if hasattr(args, 'dropouts') and args.dropouts:
        C.dropouts = args.dropouts
    if hasattr(args, 'kv_bits') and args.kv_bits:
        C.kv_bits = args.kv_bits

# Set derived values
C.num_bits_list = C.a_bits
C.loss_scale = [1.0 / len(C.a_bits)] * len(C.a_bits)
