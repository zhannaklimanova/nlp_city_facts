import nltk
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer


# Function to check if a word is valid based on WordNet dictionary
def is_valid_word(word):
    return bool(wordnet.synsets(word))  # Returns True if the word exists in WordNet, False otherwise

# Function to handle hyphenated words based on whether the components are valid words
def handle_hyphenated_word(word):
    if '-' in word:
        parts = word.split('-')
        # If all parts are valid words, split them
        if all(is_valid_word(part) for part in parts):
            return parts  # Return individual parts
        else:
            return [word]  # Keep the whole hyphenated word together
    return [word]  # Return the word as it is if there's no hyphen

# Main function to tokenize and process the text
def custom_tokenize(text):
    # Tokenize the text using RegexpTokenizer to include hyphenated words
    tokenizer = RegexpTokenizer(r'\w+(-\w+)*')
    tokens = tokenizer.tokenize(text)

    # Process each token and handle hyphenated words
    processed_tokens = []
    for token in tokens:
        processed_tokens.extend(handle_hyphenated_word(token))

    return processed_tokens

# Example usage
text = "This is Verrazzano-Narrows state-of-the-art solution for self-driving cars, and it is well-known."
tokens = custom_tokenize(text)
print(tokens)
