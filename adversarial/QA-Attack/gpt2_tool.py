import time
from nltk.corpus import stopwords
import nltk
import string
import re
import torch

# Ensure stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

punctuation = set(string.punctuation)

# Some tool functions.
translator = str.maketrans('', '', string.punctuation)

def save_two_lists_to_file(list1, list2, filename):
    with open(filename, 'w') as file:
        file.write("List 1:\n")
        for line in list1:
            file.write(line + '\n')
        file.write("\nList 2:\n")
        for line in list2:
            file.write(line + '\n')

def save_show_result_to_file(show_result, filename):
    with open(filename, 'w') as file:
        for item in show_result:
            file.write(f"Question: {item[0]}\n")
            file.write(f"Context: {item[1]}\n")
            file.write(f"Adversary: {item[2]}\n")
            file.write(f"Attacked Answers: {item[3]}\n")
            file.write(f"Raw Answer: {item[4]}\n")
            file.write("-" * 50 + "\n")

def convert_seconds_to_hms(seconds):
    """Convert seconds to hours:minutes:seconds format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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

if __name__ == "__main__":
    # Test the functions
    test_result = [
        ["What is the capital of France?", "Paris is the capital of France.", "Paris is the capital of Germany.", "Berlin", "Paris"]
    ]
    save_show_result_to_file(test_result, "test_output.txt")
    print("Test completed successfully!")

