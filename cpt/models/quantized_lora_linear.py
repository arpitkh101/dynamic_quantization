"""
LoRA Modules for Switch Precision Training

This module provides a new structure for LoRA (Low-Rank Adaptation) modules that work with
quantization configurations. The main class is QuantizedLoRALinear which combines:

1. QuantConfig: Defines bit-widths for different layers and components
2. LoRAConfig: Defines LoRA adapter configurations for different bit-widths
3. QuantizedLoRALinear: The main module that uses both configs

Key Features:
- Automatic bit-width extraction from layer names and quant_config
- Multiple LoRA adapters for different bit-widths
- Configurable LoRA parameters (rank, alpha, dropout)
- Utility functions for creating configurations

Usage:
    # Create configurations
    quant_config = create_default_quant_config(num_layers=12)
    lora_config = create_default_lora_config(quant_bits=[4, 8])
    
    # Create module
    module = QuantizedLoRALinear(
        in_features=768, 
        out_features=2304,
        quant_config=quant_config,
        lora_config=lora_config
    )
    
    # Set layer name (done externally by model factory)
    module.set_layer_name("transformer.h.0.attn.c_attn")
    
    # Forward pass
    output = module(input_tensor)
"""

import scipy as sp
import torch
import torch.nn as nn
from models.utils_quant import QuantizeLinear
import math


class QuantizeLoraLinear(QuantizeLinear):
    """Enhanced QuantizeLinear with LoRA support."""
    def __init__(
        self,
        in_features, out_features,
        w_bits: int,
        a_bits: int,
        quant_config: dict = None,
        bias: bool = False,
        ranks: int = None,
        alphas: int = None,
        dropouts: int = None,
    ):
        super().__init__(in_features, out_features, bias=True)
        
        # Store dimensions
        self.in_features = in_features
        self.out_features = out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.ranks = ranks
        self.alphas = alphas
        self.dropouts = dropouts
        
        # CRITICAL FIX: Initialize layer_name
        self.layer_name = None
        
        # Configuration
        self.quant_config = quant_config if quant_config is not None else None
    
        
        self.lora_adapters = nn.ModuleDict({
                'lora_A': nn.Linear(self.in_features, self.ranks, bias=False),
                'lora_B': nn.Linear(self.ranks, self.out_features, bias=False),
                'dropout': nn.Dropout(self.dropouts)
            })
        self.lora_scaling = self.alphas / self.ranks
 
    def forward(self, x, a_bit, w_bit, quant_config=None):
        """Forward pass with quantization and LoRA."""
        # Determine quantization bit width

        # if quant_config is not None:
        #     w_bit = quant_config['w_bit'][self.layer_name]
        #     # if w_bit != 16:
        #     #     print(f"Layer_name: {self.layer_name}, w_bit: {w_bit}")
        #     a_bit = quant_config['a_bit'][self.layer_name]
            # if a_bit != 16:
            #     print(f"Layer_name: {self.layer_name}, a_bit: {a_bit}")
            #print(f"a_bit: {a_bit}")

        # Base forward pass with quantization
        output = super().forward(x, w_bits=w_bit, a_bits=a_bit)
        
        adapter = self.lora_adapters
            
            # LoRA forward pass: x -> lora_A -> dropout -> lora_B
        lora_a_out = adapter['lora_A'](x)
        lora_output = adapter['lora_B'](adapter['dropout'](lora_a_out))
        output = output + lora_output * self.lora_scaling

        return output
    
    def set_layer_name(self, layer_name: str):
        """Set the layer name for this module."""
        self.layer_name = layer_name