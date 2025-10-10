import torch
import torch.nn.functional as F
import numpy as np
from gpt2_victim_model import GPT2QAVictimModel

class HotFlipAttack:
    """
    HotFlip adversarial attack for GPT-2 QA model
    Adapted for question-answering tasks
    """
    
    def __init__(self, model_path, max_changes=5, beam_size=10):
        self.model = GPT2QAVictimModel(model_path)
        self.max_changes = max_changes
        self.beam_size = beam_size
        self.vocab = self.model.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def get_character_gradients(self, question, context):
        """
        Get gradients for each character in the context
        """
        # Tokenize input
        inputs = self.model.tokenizer(question, context, return_tensors='pt')
        input_ids = inputs['input_ids']
        
        # Get embeddings
        embeddings = self.model.model.transformer.wte(input_ids)
        embeddings.requires_grad = True
        
        # Forward pass with embeddings
        outputs = self.model.model(inputs_embeds=embeddings)
        logits = outputs.logits
        
        # Get original answer
        original_answer = self.model.get_answer(question, context)
        target_ids = self.model.tokenizer.encode(original_answer, return_tensors='pt')
        
        # Calculate loss
        loss = F.cross_entropy(logits, target_ids)
        
        # Get gradients
        loss.backward()
        gradients = embeddings.grad.data
        
        return gradients, input_ids, embeddings
    
    def find_best_character_flips(self, gradients, input_ids, context):
        """
        Find best character flips based on gradients
        """
        # Convert input_ids to text
        text = self.model.tokenizer.decode(input_ids[0])
        
        # Find context part (after question)
        question_end = text.find(context)
        context_start = question_end
        context_end = len(text)
        
        best_flips = []
        
        # For each token in context
        for i, token_id in enumerate(input_ids[0]):
            if i < context_start:
                continue
                
            # Get gradient for this token (sum over embedding dimension)
            token_gradient = torch.norm(gradients[0][i]).item()
            
            # Get original token
            original_token = self.inv_vocab.get(token_id.item(), '<UNK>')
            
            # Try flipping to similar characters
            for char in 'abcdefghijklmnopqrstuvwxyz':
                if char != original_token.lower():
                    # Calculate flip score
                    flip_score = token_gradient
                    best_flips.append((i, original_token, char, flip_score))
        
        # Sort by gradient magnitude
        best_flips.sort(key=lambda x: abs(x[3]), reverse=True)
        return best_flips[:self.beam_size]
    
    def apply_character_flip(self, text, position, old_char, new_char):
        """
        Apply a character flip to the text
        """
        # Find the character position in the text
        char_pos = 0
        for i, char in enumerate(text):
            if char_pos == position:
                return text[:i] + new_char + text[i+1:]
            char_pos += 1
        return text
    
    def beam_search_perturbations(self, question, context, max_changes=5):
        """
        Use beam search to find optimal perturbations
        """
        # Get initial gradients
        gradients, input_ids, embeddings = self.get_character_gradients(question, context)
        
        # Initialize beam with original text
        beam = [(context, 0, [])]  # (text, score, changes)
        
        for step in range(max_changes):
            candidates = []
            
            for text, score, changes in beam:
                # Find best flips for current text
                best_flips = self.find_best_character_flips(gradients, input_ids, text)
                
                for pos, old_char, new_char, flip_score in best_flips:
                    # Apply flip
                    new_text = self.apply_character_flip(text, pos, old_char, new_char)
                    
                    # Calculate new score
                    new_score = score + flip_score
                    new_changes = changes + [(pos, old_char, new_char)]
                    
                    candidates.append((new_text, new_score, new_changes))
            
            # Keep top beam_size candidates
            candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            beam = candidates[:self.beam_size]
        
        # Return best candidate
        if beam:
            return beam[0]
        return (context, 0, [])
    
    def attack_sample(self, question, context):
        """
        Attack a single sample using HotFlip
        """
        # Get original answer
        original_answer = self.model.get_answer(question, context)
        
        # Generate adversarial text
        adversarial_text, score, changes = self.beam_search_perturbations(
            question, context, self.max_changes
        )
        
        # Get adversarial answer
        adversarial_answer = self.model.get_answer(question, adversarial_text)
        
        return {
            'original_answer': original_answer,
            'adversarial_answer': adversarial_answer,
            'adversarial_text': adversarial_text,
            'changes': changes,
            'score': score,
            'success': original_answer != adversarial_answer
        }

# Example usage
if __name__ == "__main__":
    model_path = '/data/arpit/code/outputs/gpt2_qa_switch_precision-20251004-125247'
    
    # Initialize HotFlip attack
    attack = HotFlipAttack(model_path, max_changes=3, beam_size=5)
    
    # Test attack
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France."
    
    print("Testing HotFlip attack...")
    result = attack.attack_sample(question, context)
    
    print(f"Original answer: {result['original_answer']}")
    print(f"Adversarial answer: {result['adversarial_answer']}")
    print(f"Attack successful: {result['success']}")
    print(f"Changes made: {result['changes']}")
    print(f"Adversarial text: {result['adversarial_text']}")
