# Switchable and Dynamic quantization

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

## Results

This repository evaluates switchable precision training, mixed-precision layer design,
cyclic precision training (CPT), and adversarial robustness for quantized GPT-2 models
fine-tuned on extractive QA (SQuAD).

### 1. Switchable Precision Training
The model is jointly fine-tuned across 4/6/8/16-bit paths using per-bit LoRA adapters
and distillation.

**Accuracy Overview**
| Bit | F1 | EM |
|-----|------|------|
| 32 (FP-LoRA) | **76.80** | **66.55** |
| 16 | 75.42 | 64.08 |
| 8 | 74.03 | 62.83 |
| 6 | 71.15 | 59.21 |
| 4 | 60.41 | 46.12 |

**Highlights**
- 6–8 bit achieves strong accuracy with major compression.
- Distillation is essential for stability across bit-paths.

### 2. Layer-Wise Sensitivity
Each layer of an 8-bit model is selectively reduced to 6-bit/4-bit.

**Key Findings**
- `attn.c_proj` is highly robust → safe to quantize.
- `attn.c_attn` + `mlp.c_fc` are most sensitive.
- Early layers (0–3) require higher precision; final layers (9–11) tolerate 4-bit.

**Optimal Mixed-Precision Config**
| Layers | attn.c_proj | attn.c_attn | mlp.c_fc | mlp.c_proj |
|--------|--------------|-------------|-----------|-------------|
| 0–3 | 4 | 6 | 6 | 6 |
| 4–9 | 6 | 8 | 6 | 8 |
| 9–11 | 4 | 4 | 4 | 4 |

Result: **71.18 F1 / 59.16 EM**  
83% of layers run at reduced precision.

### 3. Cyclic Precision Training (CPT)
Two variants tested:  
1) Separate LoRA adapters → unstable  
2) Shared LoRA adapters → strong + consistent

**Shared-Adapter CPT Results**
| Bit | F1 | EM |
|-----|------|------|
| 8 | 73.97 | 62.90 |
| 7 | 73.80 | 62.73 |
| 6 | **72.72** | **61.21** |
| 5 | 69.65 | 56.88 |
| 4 | 53.29 | 38.41 |

Moderate precision (6-bit) benefits most from CPT, acting as a regularizer.

### 4. Adversarial Robustness
Attacks generated using **QA-Attack** across 8-bit and 4-bit targets (500 samples).
Random precision switching is tested as a defense.

| Target | Defense | F1 Drop | EM Drop |
|--------|----------|---------|----------|
| 8-bit | Fixed | 10.57% | 15.06% |
| 8-bit | Random | **8.11%** | **7.18%** |
| 4-bit | Fixed | 16.35% | 25.61% |
| 4-bit | Random | **10.32%** | **15.31%** |

**Takeaway:** Random precision switching reduces attack effectiveness by 6–10 points
and aligns with “moving-target” defenses from Double-Win Quant.

### Summary
- Switchable precision enables multi-bit fine-tuning with minimal loss at 6–8 bits.
- Layer sensitivity reveals where aggressive quantization is safe.
- CPT improves moderate-bit accuracy when LoRA parameters are shared.
- Random precision switching provides meaningful adversarial robustness.

## References

This work builds upon several key papers in the field of quantization and adversarial robustness:

### Quantization and Training
- **LLM-QAT**: Data-Free Quantization Aware Training for Large Language Models
- **InstantNet**: Automated Generation and Deployment of Instantaneously Switchable-Precision Networks
- **CPT**: Efficient Deep Neural Network Training via Cyclic Precision
- **Double Win Quant**: Aggressively Winning Robustness of Quantized Deep Neural Networks via Random Precision Training and Inference

### Adversarial Attacks
- **QA Attack**: Deceiving Question-Answering Models: A Hybrid Word-Level Adversarial Approach

