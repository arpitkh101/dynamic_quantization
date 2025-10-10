import re
from bert_mlm import BertMLMGuesser

def mask_and_predict(sentence, word_to_attack, num_of_predict=5, single=False):
    """
    Mask words in the sentence and get predictions from BERT MLM.
    
    Args:
        sentence: The input sentence
        word_to_attack: List of tuples (position, word) to attack
        num_of_predict: Number of predictions to get for each masked word
        single: If True, return only the top prediction
    
    Returns:
        List of predictions for each masked word
    """
    guesser = BertMLMGuesser()
    predictions = []
    
    for position, word in word_to_attack:
        # Create masked sentence
        masked_sentence = sentence[:position] + "[MASK]" + sentence[position + len(word):]
        
        # Get predictions
        predicted_tokens = guesser.guess_masked_token(masked_sentence, num_of_predict)
        
        if single:
            predictions.append([predicted_tokens[0]])  # Only top prediction
        else:
            predictions.append(predicted_tokens)
    
    return predictions

def generate_candidate_sentences(sentence, word_to_attack, candidates, single=False):
    """
    Generate candidate sentences by replacing words with predictions.
    
    Args:
        sentence: The original sentence
        word_to_attack: List of tuples (position, word) to attack
        candidates: List of predictions for each word
        single: If True, use only the first candidate for each word
    
    Returns:
        List of candidate sentences
    """
    candidate_sentences = []
    
    if single:
        # Generate one candidate sentence using top predictions
        candidate_sentence = sentence
        for i, (position, word) in enumerate(word_to_attack):
            if i < len(candidates) and len(candidates[i]) > 0:
                replacement = candidates[i][0]
                candidate_sentence = candidate_sentence[:position] + replacement + candidate_sentence[position + len(word):]
        candidate_sentences.append(candidate_sentence)
    else:
        # Generate multiple candidate sentences
        for i, (position, word) in enumerate(word_to_attack):
            if i < len(candidates):
                for candidate in candidates[i]:
                    candidate_sentence = sentence[:position] + candidate + sentence[position + len(word):]
                    candidate_sentences.append(candidate_sentence)
    
    return candidate_sentences

def count_unique_words_in_text1(text):
    """
    Count unique words in the given text.
    
    Args:
        text: Input text string
    
    Returns:
        Number of unique words
    """
    # Simple word tokenization
    words = re.findall(r'\b\w+\b', text.lower())
    return len(set(words))

if __name__ == "__main__":
    # Test the functions
    sentence = "The capital of France is Paris."
    word_to_attack = [("Paris", 25)]  # Position of "Paris" in the sentence
    
    predictions = mask_and_predict(sentence, word_to_attack, num_of_predict=3)
    print(f"Predictions: {predictions}")
    
    candidates = generate_candidate_sentences(sentence, word_to_attack, predictions)
    print(f"Candidate sentences: {candidates}")
    
    word_count = count_unique_words_in_text1(sentence)
    print(f"Unique words: {word_count}")
