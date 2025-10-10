# Switchable and Dynamic quantization for GPT-2 Question Answering

This project implements switch precision training for GPT-2 models on Question Answering tasks, combining dynamic quantization techniques with LoRA (Low-Rank Adaptation) for efficient training and inference.

## Overview

The project focuses on training GPT-2 models with dynamic precision switching during training, allowing the model to learn optimal quantization strategies for different layers and components. This approach enables significant model compression while maintaining performance on QA tasks.

## Key Features

- **Switch Precision Training**: Dynamic switching between different user specified bit-widths (eg. 3, 4, 6, 8, 16 bits) during training
- **LoRA Integration**: Low-Rank Adaptation for Efficient Fine-Tuning of Quantized Models with Support for Bit-Width-Specific Adapters.
- **Knowledge Distillation**: Teacher-student training with cascaded distillation
- **Layer-wise Sensitivity Analysis**: Comprehensive analysis of quantization sensitivity across different model layers
- **Cyclic Precision Training**: Periodic cycling between different quantization levels during training for improved robustness
- **Adversarial Robustness**: Comprehensive evaluation of quantized models against adversarial attacks


## Project Structure

```
├── train.py                          # Main training script
├── config_new.py                     # Configuration management
├── layer_sensitivity.py              # Layer sensitivity analysis
├── evaluate_optimal_config.py        # Evaluation of optimal quantization configs
├── models/                           # Custom model implementations
│   ├── modeling_gpt2_quant.py       # Quantized GPT-2 model
│   ├── quantized_lora_linear.py     # LoRA with quantization
│   └── configuration_gpt2.py        # GPT-2 configuration
├── utils.py                          # Utility functions
├── training_utils.py                 # Training utilities
├── adversarial/                      # Adversarial attack evaluation
│   └── QA-Attack/                   # QA-specific attack implementations
        ├── gpt2_attack.py           # Main adversarial attack framework
        ├── gpt2_victim_model.py     # GPT-2 victim model wrapper
        ├── gpt2_ranking.py          # Attention and removal based ranking attacks
        ├── evaluate_comprehensive_adversarial_dataset.py  # Large-scale evaluation
        └── create_comprehensive_adversarial_dataset.py    # Dataset generation
└── cpt/                             # Cyclic precision training experiments
    ├── train_cyclic.py              # Cyclic precision training implementation
    └── train_cyclic.sh              # Training script for cyclic precision
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install torch transformers datasets evaluate tensorboardX easydict tqdm matplotlib seaborn pandas numpy
```

## Usage

### Training

Basic training with switch precision:
```bash
python train.py \
    --dataset_name squad \
    --model_name_or_path gpt2 \
    --a_bits 4 6 8 \
    --w_bits 4 6 8 \
    --ranks 16 8 8 \
    --alphas 32.0 16.0 16.0 \
    --dropouts 0.05 0.01 0.01 \
    --learning_rate 1e-3 \
    --max_steps 1000 \
    --do_train \
    --per_device_train_batch_size 16
```

### Cyclic Precision Training

Training with periodic precision cycling:
```bash
python cpt/train_cyclic.py \
    --is_cyclic_precision \
    --cyclic_a_bits_schedule 4 8 \
    --cyclic_w_bits_schedule 4 8 \
    --num_cyclic_period 32 \
    --max_steps 1000 \
    --do_train \
    --learning_rate 1e-3 \
    --ranks 16 \
    --alphas 32.0
```

### Layer Sensitivity Analysis

Analyze quantization sensitivity across model layers:
```bash
python layer_sensitivity.py \
    --pretrained_model_path /path/to/checkpoint.pt \
    --test_bits 4 6 \
    --original_a_bits 4 6 8 16 \
    --original_w_bits 4 6 8 16
```

### Optimal Configuration Evaluation

Evaluate the optimal mixed quantization configuration (may need to edit the file for exact config details):
```bash
python evaluate_optimal_config.py
```

### Adversarial Evaluation

Run comprehensive adversarial attacks on trained models:
```bash
# Basic adversarial attack
python adversarial/QA-Attack/gpt2_attack.py \
    --model_path /path/to/checkpoint.pt \
    --dataset_name squad \
    --n 10 \
    --attack_words 3


# Create adversarial dataset with hybrid attacks
python adversarial/QA-Attack/create_comprehensive_adversarial_dataset.py \
    --model_path /path/to/checkpoint.pt \
    --num_samples 500 \
    --attack_words 3 \
    --output_file /path/to/dataset.json

# Evaluate comprehensive adversarial dataset
python adversarial/QA-Attack/evaluate_comprehensive_adversarial_dataset.py \
    --model_path /path/to/checkpoint.pt \
    --dataset_file /path/to/dataset.json
```

## Configuration

The project uses a comprehensive configuration system (`config_new.py`) that supports:

- **Quantization Parameters**: Bit-widths for activations and weights
- **LoRA Settings**: Ranks, alphas, and dropout rates
- **Training Hyperparameters**: Learning rates, batch sizes, schedules
- **Evaluation Settings**: Metrics, logging, and checkpointing

## Key Components

### Switch Precision Training

The training process dynamically switches between different precision levels:
- **avg_loss**: Average loss across all bit-widths
- **cascaded**: Progressive distillation from high to low precision
- **Knowledge Distillation**: Teacher-student training with temperature scaling

### Cyclic Precision Training

Periodic cycling between different precision levels:
- **Cyclic Schedule**: Alternates between different bit-widths during training
- **Period Control**: Configurable cycling frequency
- **Robustness**: Improves model robustness to quantization

### Quantized LoRA Linear Layers

Custom implementation combining:
- Dynamic quantization with configurable bit-widths
- LoRA adaptation for efficient fine-tuning
- Layer-wise precision control

### Layer Sensitivity Analysis

Comprehensive analysis framework that:
- Tests individual layer quantization sensitivity
- Generates heatmaps and visualizations



### Adversarial Evaluation

#### Attack Strategies
- **Attention-based Ranking**: Uses model attention weights to identify critical words for perturbation
- **Word Removal Attacks**: Systematically removes important words to test model robustness
- **Combined Ranking**: Integrates multiple ranking strategies for enhanced attack effectiveness

#### Evaluation Framework
- **Victim Model Wrapper**: Specialized GPT-2 QA model wrapper for adversarial testing
- **Multi-metric Evaluation**: F1 score, Exact Match, and attack success rate analysis
- **BERT-based Candidate Generation**: Uses BERT MLM for generating adversarial word candidates

## References

This work builds upon several key papers in the field of quantization and adversarial robustness:

### Quantization and Training
- **LLM-QAT**: Data-Free Quantization Aware Training for Large Language Models
- **Instant Net**: Automated Generation and Deployment of Instantaneously Switchable-Precision Networks
- **CPT**: Efficient Deep Neural Network Training via Cyclic Precision
- **Double Win Quant**: Aggressively Winning Robustness of Quantized Deep Neural Networks via Random Precision Training and Inference

### Adversarial Attacks
- **QA Attack**: Deceiving Question-Answering Models: A Hybrid Word-Level Adversarial Approach

