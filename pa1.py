import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words
from sklearn.feature_extraction.text import CountVectorizer
import string
import contractions
import re
import inflect
from collections import Counter
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from itertools import chain


stemmer = PorterStemmer()
inflect_engine = inflect.engine()

stop_words = set(stopwords.words('english'))




# ============================================================
# ===================== PRE-PROCESSING =======================
# ============================================================

def preprocess(text):
    # Step 1: Lowercase the text and expand contractions
    text = text.lower()
    text = contractions.fix(text)

    # Step 2: Remove 's and ’s from possessives (e.g., "city's" -> "city" and "city’s" -> "city")
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)

    # Step 3: Remove all punctuation except for hyphens and periods between numbers
    text = re.sub(r"[^\w\s\-.]", "", text)  # Keep hyphens and periods
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)  # Remove periods not between numbers

    # Step 4: Remove periods followed by a space
    text = re.sub(r"\.\s", " ", text)

    # Step 5: Remove trailing period (if it's at the end of the text)
    text = re.sub(r"\.$", "", text)  # Remove period if it is at the very end of the text

    # Step 6: Remove extra spaces (if punctuation removal leaves double spaces)
    text = re.sub(r'\s+', ' ', text).strip()  # Ensures no extra spaces

    # Step 7: Tokenize the text while keeping decimal numbers and hyphenated words intact
    tokenizer = RegexpTokenizer(r'\d+\.\d+|\w+(?:-\w+)*')
    words = tokenizer.tokenize(text)

    # Step 8: Convert plural words to singular
    singular_words = [inflect_engine.singular_noun(word) if inflect_engine.singular_noun(word) else word for word in words]

    # Step 9: Remove stop words
    filtered_words = [word for word in singular_words if word not in stop_words]

    return filtered_words

def get_vocabulary_list(lists_of_words):
    # Flatten the list of lists into a single list of words
    flat_list = list(chain.from_iterable(lists_of_words))

    # Create a set from the flat list to remove duplicates
    vocabulary = set(flat_list)

    # Return the sorted vocabulary list
    return sorted(vocabulary)

def extend_vocabulary_with_nltk(vocabulary_list):
    # Get the list of English words from NLTK
    english_words = set(words.words())

    # Combine your existing vocabulary list with the NLTK English words
    combined_vocabulary = set(vocabulary_list) | english_words  # Union of both sets

    # Return the sorted combined vocabulary list
    return sorted(combined_vocabulary)

def get_feature_matrix(processed_facts, vocabulary_list):
    # Create a word-to-index mapping from the vocabulary list
    word_to_index = {word: idx for idx, word in enumerate(vocabulary_list)}

    # Initialize an empty matrix where rows = number of facts, columns = vocabulary size
    feature_matrix = np.zeros((len(processed_facts), len(vocabulary_list)), dtype=int)

    # For each fact, calculate the frequency of each word
    for i, fact in enumerate(processed_facts):
        word_count = Counter(fact)  # Get word frequencies for this fact
        for word, count in word_count.items():
            if word in word_to_index:
                column_index = word_to_index[word]
                feature_matrix[i, column_index] = count

    return feature_matrix

def create_labels(num_facts, label_value):
    # Create a numpy array of shape (num_facts, 1) filled with label_value
    labels = np.full((num_facts, 1), label_value, dtype=int)
    return labels

def combine_facts_and_labels(real_facts_matrix, fake_facts_matrix, real_labels, fake_labels):

    # Combine real and fake facts into one matrix X (200, 712)
    X = np.vstack((real_facts_matrix, fake_facts_matrix))

    # Combine real and fake labels into one array y (200, 1)
    y = np.vstack((real_labels, fake_labels))

    return X, y

def train_and_evaluate(X_train, y_train, X_valid, y_valid):
    # Initialize the classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC()
    }

    best_classifier = None
    best_accuracy = 0

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train.ravel())  # Train the classifier

        # Predict on the validation set
        y_valid_pred = clf.predict(X_valid)

        # Calculate validation accuracy
        accuracy = np.sum(y_valid_pred == y_valid.ravel()) / len(y_valid)

        print(f"{name} Validation Accuracy: {accuracy:.4f}")

        # Update the best classifier based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier = clf

    return best_classifier, best_accuracy

if __name__ == "__main__":
    # Load real and fake facts
    with open('facts.txt', 'r') as real_facts_file, open('fakes.txt', 'r') as fake_facts_file:
        real_facts = real_facts_file.readlines()
        fake_facts = fake_facts_file.readlines()

    # Calculate the number of real and fake facts using len()
    num_real_facts = len(real_facts)
    num_fake_facts = len(fake_facts)

    # Process facts using the preprocess function
    processed_real_facts = list(map(preprocess, real_facts))
    processed_fake_facts = list(map(preprocess, fake_facts))

    # Combine both processed_real_facts and processed_fake_facts into a vocabulary list
    combined_facts = processed_real_facts + processed_fake_facts
    vocabulary_list = get_vocabulary_list(combined_facts)

    # Extend the vocabulary list with the NLTK English words
    extended_vocabulary_list = extend_vocabulary_with_nltk(vocabulary_list)

    print(len(extended_vocabulary_list))
    # print(len(vocabulary_list))

    # Create feature matrices for real and fake facts
    real_facts_matrix = get_feature_matrix(processed_real_facts, set(words.words())) # here should be something like extended_vocabulary_list or vocabulary_list
    fake_facts_matrix = get_feature_matrix(processed_fake_facts, set(words.words()))

    # print("Real Facts Matrix Shape:", real_facts_matrix.shape)
    # print("Fake Facts Matrix Shape:", fake_facts_matrix.shape)

    # # Optionally, print the matrices
    # print("\nReal Facts Matrix:\n", real_facts_matrix)
    # print("\nFake Facts Matrix:\n", fake_facts_matrix)

    # print(real_facts_matrix.shape)


   # Create labels for real and fake facts
    real_labels = create_labels(num_real_facts, 1)  # 1 for real facts
    fake_labels = create_labels(num_fake_facts, 0)  # 0 for fake facts

    # print("Real Facts Matrix Shape:", real_facts_matrix.shape)
    # print("Fake Facts Matrix Shape:", fake_facts_matrix.shape)
    # print("Real Labels Shape:", real_labels.shape)
    # print("Fake Labels Shape:", fake_labels.shape)

    # Combine facts and labels into X and y
    X, y = combine_facts_and_labels(real_facts_matrix, fake_facts_matrix, real_labels, fake_labels)

    # Check the shapes of the resulting arrays
    # print("X Shape:", X.shape)  # Should be (200, 712)
    # print("y Shape:", y.shape)  # Should be (200, 1)

    # # Optionally, print the combined X and y
    # print("\nX (Combined Feature Matrix):\n", X)
    # print("\nY (Combined Labels):\n", y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(X_train.shape)

    # Train and evaluate classifiers using the validation set
    best_classifier, best_accuracy = train_and_evaluate(X_train, y_train, X_valid, y_valid)

    print(f"Best Classifier: {best_classifier.__class__.__name__} with accuracy: {best_accuracy:.4f}")

    # Evaluate the best classifier on the test set
    y_test_pred = best_classifier.predict(X_test)
    test_accuracy = np.sum(y_test_pred == y_test.ravel()) / len(y_test)

    print(f"Test Accuracy of Best Classifier: {test_accuracy:.4f}")