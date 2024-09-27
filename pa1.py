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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from itertools import chain
from nltk.util import ngrams  # For bigrams and trigrams

# Initialize global tools
stemmer = PorterStemmer()
inflect_engine = inflect.engine()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"\b(\w+)[’']s\b", r"\1", text)
    text = re.sub(r"[^\w\s\-.]", "", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    text = re.sub(r"\.\s", " ", text)
    text = re.sub(r"\.$", "", text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokenizer = RegexpTokenizer(r'\d+\.\d+|\w+(?:-\w+)*')
    words = tokenizer.tokenize(text)

    # Singularize the words and remove stop words
    singular_words = [inflect_engine.singular_noun(word) if inflect_engine.singular_noun(word) else word for word in words]
    filtered_words = [word for word in singular_words if word not in stop_words]

    return filtered_words

def get_ngrams(texts, n):
    ngram_counter = Counter()
    for text in texts:
        ngram_counter.update(ngrams(text, n))
    return ngram_counter

def get_top_ngrams(ngram_counter, top_n):
    return dict(ngram_counter.most_common(top_n))

def get_vocabulary_list(lists_of_words, bigrams, trigrams):
    flat_list = list(chain.from_iterable(lists_of_words))

    # Convert bigrams and trigrams to string form
    bigrams_str = [' '.join(bigram) for bigram in bigrams]
    trigrams_str = [' '.join(trigram) for trigram in trigrams]

    # Add bigrams and trigrams to vocabulary
    vocabulary = set(flat_list + bigrams_str + trigrams_str)

    return sorted(vocabulary)

def extend_vocabulary_with_nltk(vocabulary_list):
    english_words = set(words.words())
    combined_vocabulary = set(vocabulary_list) | english_words
    return sorted(combined_vocabulary)

def get_feature_matrix(processed_facts, vocabulary_list, bigrams, trigrams):
    bigrams_str = [' '.join(bigram) for bigram in bigrams]
    trigrams_str = [' '.join(trigram) for trigram in trigrams]

    # Create a word-to-index mapping from the vocabulary list
    word_to_index = {word: idx for idx, word in enumerate(vocabulary_list)}

    # Initialize an empty matrix where rows = number of facts, columns = vocabulary size
    feature_matrix = np.zeros((len(processed_facts), len(vocabulary_list)), dtype=int)

    for i, fact in enumerate(processed_facts):
        word_count = Counter(fact)
        bigram_count = Counter(ngrams(fact, 2))
        trigram_count = Counter(ngrams(fact, 3))

        # Update the feature matrix with word frequencies
        for word, count in word_count.items():
            if word in word_to_index:
                column_index = word_to_index[word]
                feature_matrix[i, column_index] = count

        # Update bigrams in the feature matrix
        for bigram, count in bigram_count.items():
            bigram_str = ' '.join(bigram)
            if bigram_str in word_to_index:
                column_index = word_to_index[bigram_str]
                feature_matrix[i, column_index] = count

        # Update trigrams in the feature matrix
        for trigram, count in trigram_count.items():
            trigram_str = ' '.join(trigram)
            if trigram_str in word_to_index:
                column_index = word_to_index[trigram_str]
                feature_matrix[i, column_index] = count

    return feature_matrix

def create_labels(num_facts, label_value):
    return np.full((num_facts, 1), label_value, dtype=int)

def combine_facts_and_labels(real_facts_matrix, fake_facts_matrix, real_labels, fake_labels):
    X = np.vstack((real_facts_matrix, fake_facts_matrix))
    y = np.vstack((real_labels, fake_labels))
    return X, y

def train_and_select_best_hyperparams(X_train, y_train, X_valid, y_valid):
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
        'SVM': {'C': [0.01, 0.1, 1, 10], 'kernel': ['linear', 'rbf']},
        'KNN': {'n_neighbors': [3, 5, 7, 9]},
    }

    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }

    best_classifiers = {}
    best_scores = {}

    for name, clf in classifiers.items():
        if name in param_grids:
            grid_search = GridSearchCV(clf, param_grids[name], cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train.ravel())
            best_clf = grid_search.best_estimator_
            best_score = grid_search.best_score_
        else:
            best_clf = clf.fit(X_train, y_train.ravel())
            y_valid_pred = best_clf.predict(X_valid)
            best_score = np.sum(y_valid_pred == y_valid.ravel()) / len(y_valid)

        print(f"{name} Best Validation Accuracy: {best_score:.4f}")
        best_classifiers[name] = best_clf
        best_scores[name] = best_score

    return best_classifiers, best_scores

def evaluate_on_test_set(classifiers, X_test, y_test):
    best_model = None
    best_test_accuracy = 0

    for name, clf in classifiers.items():
        y_test_pred = clf.predict(X_test)
        test_accuracy = np.sum(y_test_pred == y_test.ravel()) / len(y_test)
        print(f"{name} Test Accuracy: {test_accuracy:.4f}")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model = clf

    return best_model, best_test_accuracy

if __name__ == "__main__":
    with open('facts.txt', 'r') as real_facts_file, open('fakes.txt', 'r') as fake_facts_file:
        real_facts = real_facts_file.readlines()
        fake_facts = fake_facts_file.readlines()

    num_real_facts = len(real_facts)
    num_fake_facts = len(fake_facts)

    processed_real_facts = list(map(preprocess, real_facts))
    processed_fake_facts = list(map(preprocess, fake_facts))

    combined_facts = processed_real_facts + processed_fake_facts

    # Extract bigrams and trigrams from the data
    bigram_counter = get_ngrams(combined_facts, 2)
    trigram_counter = get_ngrams(combined_facts, 3)

    # Get the top 50 bigrams and trigrams (adjust the number as needed)
    top_bigrams = get_top_ngrams(bigram_counter, 50)
    top_trigrams = get_top_ngrams(trigram_counter, 50)

    # Create vocabulary list that includes top bigrams and trigrams
    vocabulary_list = get_vocabulary_list(combined_facts, top_bigrams.keys(), top_trigrams.keys())

    # Extend the vocabulary list with the NLTK English words
    extended_vocabulary_list = extend_vocabulary_with_nltk(vocabulary_list)

    # Create feature matrices for real and fake facts, including bigrams and trigrams
    real_facts_matrix = get_feature_matrix(processed_real_facts, vocabulary_list, top_bigrams.keys(), top_trigrams.keys())
    fake_facts_matrix = get_feature_matrix(processed_fake_facts, vocabulary_list, top_bigrams.keys(), top_trigrams.keys())

    real_labels = create_labels(num_real_facts, 1)
    fake_labels = create_labels(num_fake_facts, 0)

    X, y = combine_facts_and_labels(real_facts_matrix, fake_facts_matrix, real_labels, fake_labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    best_classifiers, best_validation_scores = train_and_select_best_hyperparams(X_train, y_train, X_valid, y_valid)

    best_model, best_test_accuracy = evaluate_on_test_set(best_classifiers, X_test, y_test)

    print(f"Best Model on Test Set: {best_model.__class__.__name__} with accuracy: {best_test_accuracy:.4f}")
