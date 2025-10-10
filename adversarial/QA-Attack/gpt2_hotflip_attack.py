import torch
import torch.nn.functional as F
import random
import time
from gpt2_victim_model import GPT2QAVictimModel
from data import DatasetLoader
from gpt2_tool import convert_seconds_to_hms

class GPT2HotFlipAttack:
    """
    HotFlip attack integrated with GPT-2 QA Attack framework
    """
    
    def __init__(self, model_path, max_changes=3, num_candidates=5):
        self.model = GPT2QAVictimModel(model_path)
        self.max_changes = max_changes
        self.num_candidates = num_candidates
        
        # Character substitution patterns
        self.char_substitutions = {
            'a': ['á', 'à', 'â', 'ä', 'ã', 'å'],
            'e': ['é', 'è', 'ê', 'ë'],
            'i': ['í', 'ì', 'î', 'ï'],
            'o': ['ó', 'ò', 'ô', 'ö', 'õ'],
            'u': ['ú', 'ù', 'û', 'ü'],
            'c': ['ç'],
            'n': ['ñ'],
            's': ['ß'],
            'y': ['ý', 'ÿ']
        }
        
        # Common typos
        self.typo_patterns = {
            'a': ['s', 'q', 'w'],
            'b': ['v', 'g', 'h'],
            'c': ['x', 'v', 'b'],
            'd': ['s', 'f', 'e'],
            'e': ['r', 'w', 'd'],
            'f': ['d', 'g', 'r'],
            'g': ['f', 'h', 't'],
            'h': ['g', 'j', 'y'],
            'i': ['u', 'o', 'k'],
            'j': ['h', 'k', 'u'],
            'k': ['j', 'l', 'i'],
            'l': ['k', 'o', 'p'],
            'm': ['n', 'j', 'k'],
            'n': ['m', 'b', 'j'],
            'o': ['i', 'p', 'l'],
            'p': ['o', 'l', ';'],
            'q': ['w', 'a', 's'],
            'r': ['e', 't', 'f'],
            's': ['a', 'd', 'w'],
            't': ['r', 'y', 'g'],
            'u': ['y', 'i', 'j'],
            'v': ['c', 'b', 'f'],
            'w': ['q', 'e', 's'],
            'x': ['z', 'c', 'd'],
            'y': ['t', 'u', 'h'],
            'z': ['x', 'a', 's']
        }
    
    def apply_character_changes(self, text, num_changes):
        """
        Apply character-level changes to the text
        """
        if num_changes == 0:
            return text, []
        
        text_list = list(text)
        changes = []
        positions = list(range(len(text_list)))
        random.shuffle(positions)
        
        for i in range(min(num_changes, len(positions))):
            pos = positions[i]
            char = text_list[pos].lower()
            
            # Try accent substitution first
            if char in self.char_substitutions:
                new_char = random.choice(self.char_substitutions[char])
                text_list[pos] = new_char
                changes.append((pos, char, new_char, 'accent'))
            # Try typo substitution
            elif char in self.typo_patterns:
                new_char = random.choice(self.typo_patterns[char])
                text_list[pos] = new_char
                changes.append((pos, char, new_char, 'typo'))
            # Try case change
            elif char.isalpha():
                new_char = char.upper() if char.islower() else char.lower()
                text_list[pos] = new_char
                changes.append((pos, char, new_char, 'case'))
        
        return ''.join(text_list), changes
    
    def attack_sample(self, question, context):
        """
        Attack a single sample using HotFlip
        """
        # Get original answer
        original_answer = self.model.get_answer(question, context, bit_width=8)
        
        best_result = {
            'original_answer': original_answer,
            'adversarial_answer': original_answer,
            'adversarial_text': context,
            'changes': [],
            'success': False
        }
        
        # Try different numbers of changes
        for num_changes in range(1, self.max_changes + 1):
            # Generate multiple candidates
            for _ in range(self.num_candidates):
                adversarial_text, changes = self.apply_character_changes(context, num_changes)
                
                # Get adversarial answer
                adversarial_answer = self.model.get_answer(question, adversarial_text, bit_width=8)
                
                # Check if attack was successful
                if adversarial_answer != original_answer:
                    return {
                        'original_answer': original_answer,
                        'adversarial_answer': adversarial_answer,
                        'adversarial_text': adversarial_text,
                        'changes': changes,
                        'success': True
                    }
                
                # Keep track of best result even if not successful
                if len(changes) > len(best_result['changes']):
                    best_result = {
                        'original_answer': original_answer,
                        'adversarial_answer': adversarial_answer,
                        'adversarial_text': adversarial_text,
                        'changes': changes,
                        'success': False
                    }
        
        return best_result

def attack_gpt2_hotflip(args):
    """
    Main attack function for HotFlip
    """
    print(f"\n -----------   Start GPT-2 HotFlip Attack   -------------\n")
    
    # Initialize attack
    attack = GPT2HotFlipAttack(
        model_path=args.model_path,
        max_changes=args.max_changes,
        num_candidates=args.num_candidates
    )
    
    # Load data
    loader = DatasetLoader(args.dataset_name)
    loader.load_dataset()
    data = loader.get_samples(num_samples=args.n, randomize=False)
    
    successful_attacks = 0
    total_samples = len(data)
    
    start_time = time.time()
    
    for i in range(total_samples):
        print(f"\n ----- Attacking No.{i+1} sample -----\n")
        
        # Get sample data
        if args.dataset_name == 'google/boolq':
            question, sentence = data[i][1][0], data[i][1][1]
        elif args.dataset_name == 'deepmind/narrativeqa':
            question, sentence = data[i][1][0], data[i][1][1]
        elif args.dataset_name == 'rajpurkar/squad' or args.dataset_name == 'rajpurkar/squad_v2':
            question, sentence, answer = data[i][1][0], data[i][1][1], data[i][2]
        else:
            question, sentence = data[i][1][0], data[i][1][1]
            answer = "N/A"
        
        print(f"Question: {question}")
        print(f"Context: {sentence}")
        
        # Perform attack
        result = attack.attack_sample(question, sentence)
        
        print(f"Original answer: {result['original_answer']}")
        print(f"Adversarial answer: {result['adversarial_answer']}")
        print(f"Attack successful: {result['success']}")
        print(f"Changes made: {result['changes']}")
        print(f"Adversarial text: {result['adversarial_text']}")
        
        if result['success']:
            successful_attacks += 1
    
    end_time = time.time()
    
    # Print results
    success_rate = successful_attacks / total_samples if total_samples > 0 else 0
    print(f"\nSuccessful attacks: {successful_attacks}, Total samples: {total_samples}, Success rate: {success_rate:.2f}")
    print(f"Finished!")
    print(f"Arguments: {vars(args)}")
    print(convert_seconds_to_hms(end_time - start_time))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-2 HotFlip Attack')
    parser.add_argument('--n', type=int, default=5, help='Number of samples to attack')
    parser.add_argument('--max_changes', type=int, default=3, help='Maximum character changes')
    parser.add_argument('--num_candidates', type=int, default=5, help='Number of candidates per change level')
    parser.add_argument('--dataset_name', type=str, default='rajpurkar/squad', help='Dataset name')
    parser.add_argument('--model_path', type=str, 
                       default='/data/arpit/code/outputs/gpt2_qa_switch_precision-20251004-125247',
                       help='Path to GPT-2 model')
    
    args = parser.parse_args()
    attack_gpt2_hotflip(args)
