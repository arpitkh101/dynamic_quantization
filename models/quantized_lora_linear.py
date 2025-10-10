"""
LoRA Modules for Switch Precision Training

This module provides a new structure for LoRA (Low-Rank Adaptation) modules that work with
quantization configurations. The main class is QuantizedLoRALinear which combines:

1. LoRAConfig: Defines LoRA adapter configurations for different bit-widths
2. QuantizedLoRALinear: The main module that uses both configs

Key Features:
- Automatic bit-width extraction from layer names and quant_config
- Multiple LoRA adapters for different bit-widths
- Configurable LoRA parameters (rank, alpha, dropout)
- Utility functions for creating configurations

Usage:
    # Create configurations
    lora_config = create_default_lora_config(quant_bits=[4, 8])
    
    # Create module
    module = QuantizedLoRALinear(
        in_features=768, 
        out_features=2304,
        lora_config=lora_config
    )

    
    # Forward pass
    output = module(input_tensor)
"""

import scipy as sp
import torch
import torch.nn as nn
from models.utils_quant import QuantizeLinear
import math
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class LoRAAdapterConfig:
    """Configuration for a LoRA adapter."""
    rank: int
    alpha: float
    dropout: float
    quant_bit: int
    
@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters."""
    num_adapters: int
    adapter_configs: List[LoRAAdapterConfig] = None

def create_lora_config(quant_bits: List[int] = [8], ranks: List[int] = [8], alphas: List[float] = [16.0], dropouts: List[float] = [0.01]):
    """Create LoRA configuration."""
    adapter_configs = []
    for quant_bit, rank, alpha, dropout in zip(quant_bits, ranks, alphas, dropouts):
        adapter = LoRAAdapterConfig(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            quant_bit=quant_bit
        )
        adapter_configs.append(adapter)
    
    return LoRAConfig(
        num_adapters=len(adapter_configs),
        adapter_configs=adapter_configs
    )


class QuantizeLoraLinear(QuantizeLinear):
    """Enhanced QuantizeLinear with LoRA support."""
    def __init__(
        self,
        in_features, out_features,
        w_bits: List[int],
        a_bits: List[int],
        quant_config: dict = None,
        bias: bool = False,
        ranks: List[int] = None,
        alphas: List[float] = None,
        dropouts: List[float] = None,
        lora_config: Optional[LoRAConfig] = None
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
        
        if lora_config is not None:
            self.lora_config = lora_config
        else:
            self.lora_config = create_lora_config(quant_bits=self.w_bits, ranks=self.ranks, alphas=self.alphas, dropouts=self.dropouts)
        
        self.lora_adapters = nn.ModuleDict()
        self.lora_scaling = {}
        
        if self.lora_config:
            self._init_lora_adapters(self.lora_config)
    
    def _init_lora_adapters(self, lora_config: LoRAConfig):
        """Initialize LoRA adapters."""
        if lora_config.adapter_configs is None:
            return
            
        for config in lora_config.adapter_configs:
            adapter_name = f"adapter_{config.quant_bit}"
            
            # LoRA A and B matrices with explicit dtype
            self.lora_adapters[adapter_name] = nn.ModuleDict({
                'lora_A': nn.Linear(self.in_features, config.rank, bias=False),
                'lora_B': nn.Linear(config.rank, self.out_features, bias=False),
                'dropout': nn.Dropout(config.dropout)
            })
            
            # Initialize LoRA weights
            nn.init.kaiming_uniform_(self.lora_adapters[adapter_name]['lora_A'].weight, a=5**0.5)
            nn.init.zeros_(self.lora_adapters[adapter_name]['lora_B'].weight)
            
            # Scaling factor
            self.lora_scaling[adapter_name] = config.alpha / config.rank
 
    def forward(self, x, a_bit, w_bit, quant_config=None):
        """Forward pass with quantization and LoRA."""
        # Determine quantization bit width

        if quant_config is not None:
            w_bit = quant_config['w_bit'][self.layer_name]
            # if w_bit != 16:
            #     print(f"Layer_name: {self.layer_name}, w_bit: {w_bit}")
            a_bit = quant_config['a_bit'][self.layer_name]
            # if a_bit != 16:
            #     print(f"Layer_name: {self.layer_name}, a_bit: {a_bit}")
            #print(f"a_bit: {a_bit}")

        # Base forward pass with quantization
        output = super().forward(x, w_bits=w_bit, a_bits=a_bit)
        
        # Apply LoRA adapter if it exists for this bit width
        adapter_name = f"adapter_{w_bit}"
        
        if adapter_name in self.lora_adapters:
            # Apply LoRA adaptation
            adapter = self.lora_adapters[adapter_name]
            
            # LoRA forward pass: x -> lora_A -> dropout -> lora_B
            lora_a_out = adapter['lora_A'](x)
            lora_output = adapter['lora_B'](adapter['dropout'](lora_a_out))
            output = output + lora_output * self.lora_scaling[adapter_name]

        return output
    
    def set_layer_name(self, layer_name: str):
        """Set the layer name for this module."""
        self.layer_name = layer_name