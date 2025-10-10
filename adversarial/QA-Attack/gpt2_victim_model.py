#!/usr/bin/env python3
"""
GPT-2 Victim Model for QA-Attack
Adapted to work with the trained quantized GPT-2 QA model
Now properly aligned with training script's data preparation and postprocessing
"""

import os
import sys
import torch
import logging
import numpy as np
import collections
from transformers import AutoTokenizer
import random

# Add the cpt directory to path
sys.path.append('/data/arpit/code/')

from models.configuration_gpt2 import GPT2Config
from models.modeling_gpt2_quant import GPT2ForQuestionAnswering as GPT2ForQuestionAnsweringQuant
from utils import postprocess_qa_predictions
from training_utils import create_data_preprocessing_functions, create_post_processing_function

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2QAVictimModel:
    """GPT-2 QA Victim Model for adversarial attacks with proper QA preprocessing"""
    
    def __init__(self, model_path='/data/arpit/code/outputs/gpt2_qa_switch_precision-20251007-130553', random_precision=False):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_precision = random_precision
        self.max_seq_length = 786
        self.doc_stride = 128
        self.pad_to_max_length = True
        self._load_model()
        self._setup_preprocessing_functions()
    
    def _load_model(self):
        """Load the trained GPT-2 QA model"""
        logger.info(f"Loading GPT-2 QA model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model configuration exactly like the training log shows
            gpt2_config = GPT2Config.from_pretrained('gpt2')
            gpt2_config.a_bits = [4, 6, 8, 16]
            gpt2_config.w_bits = [4, 6, 8, 16]
            gpt2_config.ranks = [16, 8, 8, 8]  
            gpt2_config.alphas = [32.0, 16.0, 16.0, 16.0]  
            gpt2_config.dropouts = [0.05, 0.01, 0.01, 0.01]
            gpt2_config.output_attentions = True
        
            
            # Initialize model
            self.model = GPT2ForQuestionAnsweringQuant(gpt2_config)
            
            # Load checkpoint
            checkpoint_path = os.path.join(self.model_path, 'final_model.pt')
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            
            # Set layer names
            for name, module in self.model.named_modules():
                if hasattr(module, 'set_layer_name'):
                    clean_name = name.replace('module.', '') if name.startswith('module.') else name
                    module.set_layer_name(clean_name)
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Successfully loaded GPT-2 QA model")
            
        except Exception as e:
            logger.error(f"Failed to load GPT-2 model: {e}")
            raise e
    
    def _setup_preprocessing_functions(self):
        """Setup preprocessing functions using the same functions as training script"""
        # Create data_args object with the necessary attributes
        class DataArgs:
            def __init__(self, max_seq_length, doc_stride, pad_to_max_length):
                self.max_seq_length = max_seq_length
                self.doc_stride = doc_stride
                self.pad_to_max_length = pad_to_max_length
                self.version_2_with_negative = False
                self.n_best_size = 20
                self.max_answer_length = 30
                self.null_score_diff_threshold = 0.0
        
        data_args = DataArgs(self.max_seq_length, self.doc_stride, self.pad_to_max_length)
        
        # Get the preprocessing functions from training script
        _, self.prepare_validation_features = create_data_preprocessing_functions(
            self.tokenizer, data_args, "question", "context", "answers"
        )
        
        # Create postprocessing function like in training script
        class TrainingArgs:
            def __init__(self):
                self.output_dir = None
        
        training_args = TrainingArgs()
        self.post_processing_function = create_post_processing_function(
            data_args, training_args, "answers", logging.INFO
        )
    
    def _prepare_qa_features(self, question, context):
        """
        Prepare QA features using the same preprocessing function as training script
        """
        # Create examples in the format expected by prepare_validation_features
        examples = {
            "id": ["example_0"],
            "question": [question],
            "context": [context],
            "answers": [{"text": [], "answer_start": []}]  # No ground truth answers needed
        }
        
        # Use the existing preprocessing function from training script
        tokenized = self.prepare_validation_features(examples)
        
        # Extract offset mapping (keep original format for postprocessing)
        offset_mapping = tokenized["offset_mapping"][0]  # Get first (and only) feature's offset mapping
        
        return tokenized, offset_mapping, None
    
    def _postprocess_qa_predictions(self, question, context, predictions, offset_mapping):
        """
        Post-process QA predictions using the same function as training script
        """
        if len(predictions) != 2:
            raise ValueError("predictions should be a tuple with two elements (start_logits, end_logits)")
        
        start_logits, end_logits = predictions
        
        # Convert to numpy if needed
        if torch.is_tensor(start_logits):
            start_logits = start_logits.cpu().numpy()
        if torch.is_tensor(end_logits):
            end_logits = end_logits.cpu().numpy()
        
        # Create the examples structure that mimics HuggingFace dataset behavior
        # It needs to support both dict-like access (examples["id"]) and list-like iteration
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            
            def __getitem__(self, key):
                if isinstance(key, str):
                    # Dict-like access: examples["id"]
                    return self.data[key]
                elif isinstance(key, int):
                    # List-like access: examples[0]
                    return {k: v[key] for k, v in self.data.items()}
                else:
                    raise TypeError(f"Invalid key type: {type(key)}")
            
            def __iter__(self):
                # Support iteration: for example in examples
                for i in range(len(self.data["id"])):
                    yield self[i]
            
            def __len__(self):
                return len(self.data["id"])
        
        examples = SimpleDataset({
            "id": ["example_0"],
            "question": [question],
            "context": [context],
            "answers": [{"text": [], "answer_start": []}]  # We don't have ground truth answers
        })
        
        features = [{
            "example_id": "example_0",
            "offset_mapping": offset_mapping
        }]
        
        # Use the direct postprocess_qa_predictions function
        predictions_dict = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=(start_logits, end_logits),
            version_2_with_negative=False,
            n_best_size=20,
            max_answer_length=30,
            null_score_diff_threshold=0.0
        )
        
        # Extract the answer for our example
        answer = predictions_dict.get("example_0", "")
        
        return answer.strip()
    
    def get_answer(self, question, context, bit_width=4):
        """Get answer from the model for a question-context pair using proper QA preprocessing"""
        # Prepare features using the same preprocessing as training
        tokenized, offset_mapping, sample_mapping = self._prepare_qa_features(question, context)
        
        # Convert to tensors and move to device
        inputs = {}
        for k, v in tokenized.items():
            if k in ['input_ids', 'attention_mask']:
                # Convert lists to tensors
                if isinstance(v, list) and len(v) > 0:
                    inputs[k] = torch.tensor(v).to(self.device)
                else:
                    inputs[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
            else:
                inputs[k] = v
        
        # Remove fields that are only needed for post-processing, not for model forward
        model_inputs = {k: v for k, v in inputs.items() if k not in ['offset_mapping', 'example_id']}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**model_inputs, a_bit=bit_width, w_bit=bit_width)
        
        # Post-process predictions to get the answer
        answer = self._postprocess_qa_predictions(
            question, context, 
            (outputs.start_logits, outputs.end_logits), 
            offset_mapping  # offset_mapping is already the first feature's offset mapping
        )
        
        return answer
    
    def get_logits(self, question, context, bit_width=4):
        """Get model logits for a question-context pair using proper QA preprocessing"""
        # Prepare features using the same preprocessing as training
        tokenized, offset_mapping, sample_mapping = self._prepare_qa_features(question, context)
        
        # Convert to tensors and move to device
        inputs = {}
        for k, v in tokenized.items():
            if k in ['input_ids', 'attention_mask']:
                # Convert lists to tensors
                if isinstance(v, list) and len(v) > 0:
                    inputs[k] = torch.tensor(v).to(self.device)
                else:
                    inputs[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
            else:
                inputs[k] = v
        
        # Remove fields that are only needed for post-processing, not for model forward
        model_inputs = {k: v for k, v in inputs.items() if k not in ['offset_mapping', 'example_id']}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**model_inputs, a_bit=bit_width, w_bit=bit_width)
        
        # Return combined logits as a single tensor for compatibility with ranking methods
        # Concatenate start and end logits to create a single tensor
        combined_logits = torch.cat([outputs.start_logits, outputs.end_logits], dim=-1)
        return combined_logits
    
    def get_attention_weights(self, question, context, bit_width=4):
        """Get attention weights from the model for a question-context pair"""
        # Prepare features using the same preprocessing as training
        tokenized, offset_mapping, sample_mapping = self._prepare_qa_features(question, context)
        
        # Convert to tensors and move to device
        inputs = {}
        for k, v in tokenized.items():
            if k in ['input_ids', 'attention_mask']:
                # Convert lists to tensors
                if isinstance(v, list) and len(v) > 0:
                    inputs[k] = torch.tensor(v).to(self.device)
                else:
                    inputs[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
            else:
                inputs[k] = v
        
        # Remove fields that are only needed for post-processing, not for model forward
        model_inputs = {k: v for k, v in inputs.items() if k not in ['offset_mapping', 'example_id']}
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**model_inputs, a_bit=bit_width, w_bit=bit_width, output_attentions=True)
        
        # Return attention weights and tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        return outputs.attentions, tokens
    
    def ask_yes_no_question(self, question, context, bit_width=8):
        """Ask a yes/no question - adapted for extractive QA"""
        answer = self.get_answer(question, context, bit_width)
        
        # Convert answer to yes/no format for compatibility
        answer_lower = answer.lower().strip()
        
        # Simple heuristics for yes/no conversion
        if any(word in answer_lower for word in ['yes', 'true', 'correct', 'right', 'indeed']):
            return "yes"
        elif any(word in answer_lower for word in ['no', 'false', 'incorrect', 'wrong', 'not']):
            return "no"
        else:
            # If we can't determine, return the actual answer
            return answer

# Global instance for compatibility with QA-Attack
gpt2_victim_model = GPT2QAVictimModel()

def ask_yes_no_question(question, context):
    """Compatibility function for QA-Attack"""
    return gpt2_victim_model.ask_yes_no_question(question, context)

def get_logits(question, context):
    """Compatibility function for QA-Attack"""
    return gpt2_victim_model.get_logits(question, context)

if __name__ == "__main__":
    # Test the model
    model = GPT2QAVictimModel()
    
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France. It is located in northern France."
    
    answer = model.get_answer(question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")
    
    # Test yes/no conversion
    yn_answer = model.ask_yes_no_question("Is Paris the capital of France?", context)
    print(f"Yes/No Answer: {yn_answer}")