import torch
import nltk
import string
from nltk.corpus import stopwords
from gpt2_victim_model import GPT2QAVictimModel

# Ensure you have downloaded the stopwords
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

punctuation = set(string.punctuation)

def clean_token(token):
    """Clean token by removing punctuation and converting to lowercase."""
    return token.lower().strip(string.punctuation)

def is_valid_token(token):
    """Check if token is valid (not stopword, not punctuation, not empty)."""
    cleaned = clean_token(token)
    return (cleaned and 
            cleaned not in stop_words and 
            cleaned not in punctuation and 
            len(cleaned) > 1)

def ranking_with_attention_gpt2(question, context, gpt2_model, top_k=5, rate=0.1, bit_width=8):
    """
    Real attention-based ranking using GPT-2 model's actual attention weights.
    """
    try:
        # Get attention weights and tokens from the model
        attentions, tokens = gpt2_model.get_attention_weights(question, context, bit_width=bit_width)
        
        # Check if attention weights are available
        if attentions is None or len(attentions) == 0:
            print("No attention weights available, falling back to heuristic")
            return ranking_with_attention_heuristic(question, context, gpt2_model, top_k, rate)
        
        # Find the start of context tokens (after question tokens)
        question_tokens = gpt2_model.tokenizer.tokenize(question)
        context_start_idx = len(question_tokens) + 2  # +2 for special tokens
        
        # Average attention across all layers and heads (following T5 approach)
        # attentions is a tuple of (batch_size, num_heads, seq_len, seq_len) for each layer
        averaged_attentions = []
        for layer_attention in attentions:
            if layer_attention is not None:
                # Average across heads: (batch_size, seq_len, seq_len)
                layer_avg = layer_attention.mean(dim=1)
                averaged_attentions.append(layer_avg)
        
        if not averaged_attentions:
            # Fallback to heuristic if no attention weights
            print("No valid attention weights found, falling back to heuristic")
            return ranking_with_attention_heuristic(question, context, gpt2_model, top_k, rate)
        
        # Apply softmax normalization to each layer (following T5 approach)
        normalized_attentions = [torch.nn.functional.softmax(layer_attention, dim=-1) for layer_attention in averaged_attentions]
        
        # Average across all layers: (batch_size, seq_len, seq_len)
        overall_attention = sum(normalized_attentions) / len(normalized_attentions)
        
        # Get attention scores for each token: (seq_len,)
        attention_scores = overall_attention[0].mean(dim=0).detach().cpu().numpy()
        
        # Apply max-score normalization (following T5 approach)
        max_score = max(attention_scores) if attention_scores.size > 0 else 1  # Avoid division by zero
        attention_scores = attention_scores / max_score
        
        # Extract context tokens and their attention scores (following T5 approach)
        context_tokens = tokens[context_start_idx:]
        context_attention_scores = attention_scores[context_start_idx:]
        
        # Create attention data similar to T5 approach
        attention_data = list(zip(context_tokens, context_attention_scores))
        
        # Map attention scores to word positions in original context
        context_words = context.split()
        context_words_clean = [clean_token(word) for word in context_words]
        
        # Filter context tokens to only include those present in context_words (following T5 approach)
        context_tokens_scored = []
        for i, (token, score) in enumerate(attention_data):
            # Clean token for comparison
            clean_token_text = token.replace('Ġ', '').replace('▁', '').lower()
            
            # Check if this token corresponds to a word in the original context
            if clean_token_text in context_words_clean and is_valid_token(clean_token_text):
                word_idx = context_words_clean.index(clean_token_text)
                context_tokens_scored.append((clean_token_text, score, word_idx))
        
        # Sort by attention score (descending) - following T5 approach
        context_tokens_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to the expected format: (score, word_idx, word, clean_word)
        valid_scored_tokens = []
        for clean_word, attention_score, word_idx in context_tokens_scored:
            original_word = context_words[word_idx]
            valid_scored_tokens.append((attention_score, word_idx, original_word, clean_word))
        
        # Return top k tokens
        if rate:
            num_tokens = max(1, int(len(valid_scored_tokens) * rate))
        else:
            num_tokens = min(top_k, len(valid_scored_tokens))
        
        return valid_scored_tokens[:num_tokens]
        
    except Exception as e:
        print(f"Error in attention-based ranking: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to heuristic method
        return ranking_with_attention_heuristic(question, context, gpt2_model, top_k, rate)

def ranking_with_attention_heuristic(question, context, gpt2_model, top_k=5, rate=0.1):
    """
    Advanced heuristic-based ranking that mimics attention-based selection.
    This approach uses multiple signals to identify important tokens.
    """
    # Tokenize the context
    tokens = context.split()
    
    # Filter valid tokens
    valid_tokens = []
    for i, token in enumerate(tokens):
        if is_valid_token(token):
            valid_tokens.append((i, token, clean_token(token)))
    
    # Get question tokens for overlap analysis
    question_tokens = set(clean_token(t) for t in question.split() if is_valid_token(t))
    
        # Get the model's answer to understand what it's focusing on
    try:
        model_answer = gpt2_model.get_answer(question, context, bit_width=8)
        answer_tokens = set(clean_token(t) for t in model_answer.split() if is_valid_token(t))
    except:
        answer_tokens = set()
    
    scored_tokens = []
    for i, token, cleaned in valid_tokens:
        score = 0
        
        # 1. Question overlap (high importance)
        if cleaned in question_tokens:
            score += 3.0
        
        # 2. Answer overlap (very high importance - tokens that appear in the answer)
        if cleaned in answer_tokens:
            score += 4.0
        
        # 3. Semantic importance (longer, more informative words)
        score += len(cleaned) * 0.2
        
        # 4. Frequency in context (words that appear multiple times are often important)
        frequency = context.lower().count(cleaned)
        score += min(frequency * 0.3, 2.0)  # Cap at 2.0
        
        # 5. Position importance (words near the beginning or end of sentences)
        # Check if token is near sentence boundaries
        context_lower = context.lower()
        token_pos = context_lower.find(cleaned)
        if token_pos != -1:
            # Check if it's near the beginning of a sentence
            before_token = context_lower[max(0, token_pos-20):token_pos]
            if any(punct in before_token for punct in ['.', '!', '?']):
                score += 0.5
            # Check if it's near the end of a sentence
            after_token = context_lower[token_pos:token_pos+20]
            if any(punct in after_token for punct in ['.', '!', '?']):
                score += 0.5
        
        # 6. Named entity and important word patterns
        if cleaned.istitle() or cleaned.isupper():  # Proper nouns
            score += 1.0
        if cleaned in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
            score -= 0.5  # Reduce score for common words
        
        # 7. Numerical and temporal importance
        if cleaned.isdigit() or any(word in cleaned for word in ['year', 'time', 'date', 'day', 'month']):
            score += 1.5
        
        scored_tokens.append((score, i, token, cleaned))
    
    # Sort by score (descending)
    scored_tokens.sort(key=lambda x: x[0], reverse=True)
    
    # Return top k tokens
    if rate:
        num_tokens = max(1, int(len(valid_tokens) * rate))
    else:
        num_tokens = min(top_k, len(scored_tokens))
    
    return scored_tokens[:num_tokens]

def ranking_with_removal_gpt2(question, context, gpt2_model, top_k=5, rate=0.1, bit_width=8):
    """
    Removal-based ranking using GPT-2 model, following the same approach as rank_with_removal.py.
    Tests removing each token and measures logit changes.
    """
    # Get baseline logits from original input
    baseline_logits = gpt2_model.get_logits(question, context, bit_width=bit_width)
    
    # Tokenize the context and apply clean_token preprocessing
    words = context.split()
    cleaned_words = [clean_token(word) for word in words]
    
    importance_scores = []
    for i, word in enumerate(words):
        if not is_valid_token(word):
            continue  # Skip the iteration if the word is not valid
        
        # Create context with token masked (following T5 approach)
        new_context = " ".join(cleaned_words[:i] + ['<mask>'] + cleaned_words[i+1:])
        
        # Get logits with modified context
        new_logits = gpt2_model.get_logits(question, new_context, bit_width=bit_width)
        
        # Measure change as the max absolute difference across all logits
        # Handle different tensor sizes by using max values
        baseline_max = torch.max(baseline_logits).item()
        new_max = torch.max(new_logits).item()
        logit_change = abs(baseline_max - new_max)
        importance_scores.append((word, logit_change, i))
    
    # Sort by logit change (descending)
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Normalize scores (following T5 approach)
    max_score = importance_scores[0][1] if importance_scores else 1
    normalized_scores = [(item[0], item[1] / max_score, item[2]) for item in importance_scores]
    
    # Return top k tokens based on rate or top_k
    if rate is not None and rate > 0:
        top_k_context_tokens = normalized_scores[:int(len(normalized_scores) * rate)]
    elif top_k is not None and top_k > 0:
        top_k_context_tokens = normalized_scores[:top_k]
    else:
        top_k_context_tokens = normalized_scores[:]
    
    # Convert to format expected by attack script: (score, position, token, cleaned_token)
    formatted_tokens = []
    for token, score, position in top_k_context_tokens:
        cleaned = clean_token(token)
        formatted_tokens.append((score, position, token, cleaned))
    
    return formatted_tokens

def combine_ranking_scores_gpt2(question, context, gpt2_model, combination='norm-link', top_k=5, rate=0.1, bit_width=8):
    """
    Combine attention and removal rankings.
    """
    attention_scores = ranking_with_attention_gpt2(question, context, gpt2_model, top_k, rate, bit_width)
    removal_scores = ranking_with_removal_gpt2(question, context, gpt2_model, top_k, rate, bit_width)
    
    # Create a dictionary to store combined scores
    combined_scores = {}
    
    # Normalize attention scores
    if attention_scores:
        max_attention = max(score for score, _, _, _ in attention_scores)
        for score, i, token, cleaned in attention_scores:
            normalized_score = score / max_attention if max_attention > 0 else 0
            combined_scores[(i, token, cleaned)] = normalized_score
    
    # Normalize removal scores
    if removal_scores:
        max_removal = max(score for score, _, _, _ in removal_scores)
        for score, i, token, cleaned in removal_scores:
            normalized_score = score / max_removal if max_removal > 0 else 0
            if (i, token, cleaned) in combined_scores:
                if combination == 'norm-add':
                    combined_scores[(i, token, cleaned)] += normalized_score
                elif combination == 'norm-link':
                    combined_scores[(i, token, cleaned)] *= normalized_score
            else:
                combined_scores[(i, token, cleaned)] = normalized_score
    
    # Sort by combined score
    sorted_tokens = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top k tokens
    if rate:
        num_tokens = max(1, int(len(combined_scores) * rate))
    else:
        num_tokens = min(top_k, len(sorted_tokens))
    
    return [(score, (i, token, cleaned)) for (i, token, cleaned), score in sorted_tokens[:num_tokens]]

if __name__ == "__main__":
    # Test the ranking functions
    gpt2_model = GPT2QAVictimModel()
    question = "What is the capital of France?"
    context = "Paris is the capital and largest city of France. It is located in northern France."
    
    print("Testing attention-based ranking:")
    attention_results = ranking_with_attention_gpt2(question, context, gpt2_model, top_k=3)
    print(attention_results)
    
    print("\nTesting removal-based ranking:")
    removal_results = ranking_with_removal_gpt2(question, context, gpt2_model, top_k=3)
    print(removal_results)
    
    print("\nTesting combined ranking:")
    combined_results = combine_ranking_scores_gpt2(question, context, gpt2_model, top_k=3)
    print(combined_results)
