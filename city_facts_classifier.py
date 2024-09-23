import contractions
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score

BINARY_FEATURES = False

def first_preprocessing(text):
    stop_words = set(stopwords.words('english'))
    preprocessed_text = text.lower() # convert to lowercase
    preprocessed_text = contractions.fix(preprocessed_text) # expand contractions
    preprocessed_text = re.sub(r"\b(\w+)[’']s\b", r"\1", preprocessed_text) # remove possessive 's or ’s
    preprocessed_text = re.sub(r"[^\w\s\-.]", "", preprocessed_text) # remove special characters except for whitespace, period, and hyphen
    preprocessed_text = re.sub(r"(?<!\d)\.(?!\d)", "", preprocessed_text) # remove period not between digits
    preprocessed_text = re.sub(r"\.\s", " ", preprocessed_text) # remove period followed by whitespace
    preprocessed_text = re.sub(r"\.$", "", preprocessed_text) # remove period at the end of the text
    preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text).strip() # remove extra whitespaces
    individual_words = RegexpTokenizer(r'\d+\.\d+|\w+(?:-\w+)*').tokenize(preprocessed_text) # split text into individual tokens
    filtered_words = [word for word in individual_words if word not in stop_words] # remove stopwords

    return filtered_words

def second_preprocessing(text_tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stemmed_words = [stemmer.stem(word) for word in text_tokens]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    return lemmatized_words

def third_preprocessing(X):
    non_zero_columns = np.any(X != 0, axis=0)
    X_processed = X[:, non_zero_columns]

    return X_processed


def count_ngram_frequency(ngrams_list, frequency_threshold=1):
    ngram_freq = dict()

    for ngrams_sublist in ngrams_list:
        for ngram in ngrams_sublist:
            ngram_tuple = tuple(ngram)
            ngram_freq[ngram_tuple] = ngram_freq.get(ngram_tuple, 0) + 1

    sorted_ngrams = sorted(ngram_freq.items(), key=lambda item: item[1], reverse=True)
    frequent_ngrams = [ngram for ngram in sorted_ngrams if ngram[1] > frequency_threshold]
    combined_frequent_ngrams = [" ".join(ngram) for ngram, _ in frequent_ngrams]

    return frequent_ngrams, combined_frequent_ngrams

def create_feature_matrix(preprocessed_facts, feature_vector, binary_features):
    feature_vector_list = list(feature_vector)
    X = np.zeros((len(preprocessed_facts), len(feature_vector_list)))

    for i, fact in enumerate(preprocessed_facts):
        for word in fact:
            if word in feature_vector_list:
                if binary_features:
                    X[i, feature_vector_list.index(word)] = 1  # use binary (presence/absence)
                else:
                    X[i, feature_vector_list.index(word)] += 1  # count frequencies

    return X

def train_and_select_best_hyperparameters(X_train, y_train, X_valid, y_valid, binary_features, use_cross_validation=False):
    if use_cross_validation:
        X_combined = np.vstack((X_train, X_valid))
        y_combined = np.vstack((y_train, y_valid)).ravel()

    results = []
    max_iterations = 10000
    k_folds = KFold(n_splits=20, shuffle=True, random_state=42)

    logistic_regression_params = {
        'C': [0.01, 0.1, 1, 5],
        'penalty': ['l1', 'l2']
    }
    if use_cross_validation:
        logistic_regression = LogisticRegression(solver='liblinear', max_iter=max_iterations)
        logistic_regression_cv = GridSearchCV(logistic_regression, logistic_regression_params, cv=k_folds)
        logistic_regression_cv.fit(X_combined, y_combined)
        best_logistic_regression_model = logistic_regression_cv.best_estimator_
        best_logistic_regression_model_params = logistic_regression_cv.best_params_
        valid_accuracy = logistic_regression_cv.best_score_
    else:
        best_logistic_regression_model_params = None
        best_validation_accuracy = 0

        for C in logistic_regression_params['C']:
            for penalty in logistic_regression_params['penalty']:
                logistic_regression = LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=max_iterations)
                logistic_regression.fit(X_train, y_train.ravel())
                y_pred = logistic_regression.predict(X_valid)
                validation_accuracy = accuracy_score(y_valid, y_pred)

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_logistic_regression_model = logistic_regression
                    best_logistic_regression_model_params = {'C': C, 'penalty': penalty}

        valid_accuracy = best_validation_accuracy

    if binary_features:
        bernoulli_naive_bayes_params = {'alpha': [0.001, 0.01, 0.1, 1.0]}
        if use_cross_validation:
            bernoulli_nb = BernoulliNB()
            grid_bayes = GridSearchCV(bernoulli_nb, bernoulli_naive_bayes_params, cv=k_folds)
            grid_bayes.fit(X_combined, y_combined)
            best_naive_bayes_model = grid_bayes.best_estimator_
            best_naive_bayes_params = grid_bayes.best_params_
            valid_accuracy = grid_bayes.best_score_
        else:
            best_naive_bayes_params = None
            best_naive_bayes_accuracy = 0
            for alpha in bernoulli_naive_bayes_params['alpha']:
                naive_bayes = BernoulliNB(alpha=alpha)
                naive_bayes.fit(X_train, y_train.ravel())
                y_pred = naive_bayes.predict(X_valid)
                valid_accuracy = accuracy_score(y_valid, y_pred)

                if valid_accuracy > best_naive_bayes_accuracy:
                    best_naive_bayes_accuracy = valid_accuracy
                    best_naive_bayes_model = naive_bayes
                    best_naive_bayes_params = {'alpha': alpha}
            valid_accuracy = best_naive_bayes_accuracy
    else:
        gaussian_naive_bayes_params = {'var_smoothing': [1e-9, 1e-7, 1e-1]}
        if use_cross_validation:
            gaussian_naive_bayes = GaussianNB()
            grid_bayes = GridSearchCV(gaussian_naive_bayes, gaussian_naive_bayes_params, cv=k_folds)
            grid_bayes.fit(X_combined, y_combined)
            best_naive_bayes_model = grid_bayes.best_estimator_
            best_naive_bayes_params = grid_bayes.best_params_
            valid_accuracy = grid_bayes.best_score_
        else:
            best_naive_bayes_params = None
            best_naive_bayes_accuracy = 0
            for var_smoothing in gaussian_naive_bayes_params['var_smoothing']:
                naive_bayes = GaussianNB(var_smoothing=var_smoothing)
                naive_bayes.fit(X_train, y_train.ravel())
                y_pred = naive_bayes.predict(X_valid)
                valid_accuracy = accuracy_score(y_valid, y_pred)

                if valid_accuracy > best_naive_bayes_accuracy:
                    best_naive_bayes_accuracy = valid_accuracy
                    best_naive_bayes_model = naive_bayes
                    best_naive_bayes_params = {'var_smoothing': var_smoothing}
            valid_accuracy = best_naive_bayes_accuracy

    perceptron_params = {'penalty': ['l1', 'l2', None], 'alpha': [0.001, 0.01, 0.1, 1]}
    if use_cross_validation:
        perceptron = Perceptron(max_iter=max_iterations)
        grid_perceptron = GridSearchCV(perceptron, perceptron_params, cv=k_folds)
        grid_perceptron.fit(X_combined, y_combined)
        best_perceptron_model = grid_perceptron.best_estimator_
        best_perceptron_params = grid_perceptron.best_params_
        valid_accuracy = grid_perceptron.best_score_
    else:
        best_perceptron_params = None
        best_validation_accuracy = 0
        for penalty in perceptron_params['penalty']:
            for alpha in perceptron_params['alpha']:
                perceptron = Perceptron(penalty=penalty, alpha=alpha, max_iter=max_iterations)
                perceptron.fit(X_train, y_train.ravel())
                y_pred = perceptron.predict(X_valid)
                valid_accuracy = accuracy_score(y_valid, y_pred)
                if valid_accuracy > best_validation_accuracy:
                    best_validation_accuracy = valid_accuracy
                    best_perceptron_model = perceptron
                    best_perceptron_params = {'penalty': penalty, 'alpha': alpha}
        valid_accuracy = best_validation_accuracy

    linear_svc_params = {
        'C': [0.001, 0.01, 0.1],
        'penalty': ['l2'],
        'loss': ['hinge', 'squared_hinge']
    }

    if use_cross_validation:
        svm = LinearSVC(max_iter=max_iterations)
        grid_svm = GridSearchCV(svm, linear_svc_params, cv=k_folds)
        grid_svm.fit(X_combined, y_combined)
        best_svm_model = grid_svm.best_estimator_
        best_svm_params = grid_svm.best_params_
        valid_accuracy = grid_svm.best_score_
    else:
        best_svm_params = None
        best_validation_accuracy = 0
        for C in linear_svc_params['C']:
            for loss in linear_svc_params['loss']:
                svm = LinearSVC(C=C, loss=loss, penalty='l2', max_iter=max_iterations)
                svm.fit(X_train, y_train.ravel())
                y_pred = svm.predict(X_valid)
                valid_accuracy = accuracy_score(y_valid, y_pred)

                if valid_accuracy > best_validation_accuracy:
                    best_validation_accuracy = valid_accuracy
                    best_svm_model = svm
                    best_svm_params = {'C': C, 'loss': loss}

    results.append((best_svm_model, best_svm_params, float(valid_accuracy)))
    results.append((best_perceptron_model, best_perceptron_params, float(valid_accuracy)))
    results.append((best_naive_bayes_model, best_naive_bayes_params, float(valid_accuracy)))
    results.append((best_logistic_regression_model, best_logistic_regression_model_params, float(valid_accuracy)))

    for model, params, valid_accuracy in results:
        print(f"Best Model: {model.__class__.__name__}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")
        print(f"Best Hyperparameters: {params}")
        print("------")

    return results

def test_best_models(results, X_test, y_test):
    test_accuracies = {}

    for model, params, valid_acc in results:
        y_pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        model_name = model.__class__.__name__
        test_accuracies[model_name] = test_accuracy

        print(f"Model: {model_name}")
        if params:
            print(f"Best Hyperparameters: {params}")
        print(f"Test Accuracy: {test_accuracy:.4f}\n")

    return test_accuracies

if __name__ == "__main__":
    with open('facts.txt', 'r') as real_facts_file, open('fakes.txt', 'r') as fake_facts_file:
        real_facts = real_facts_file.readlines()
        fake_facts = fake_facts_file.readlines()

    ### PREPROCESS ###
    feature_vector = set()
    num_real_facts = len(real_facts)
    num_fake_facts = len(fake_facts)
    real_labels = np.full((num_real_facts, 1), 1, dtype=int)
    fake_labels = np.full((num_fake_facts, 1), 0, dtype=int)

    preprocessed_real_facts = list(map(first_preprocessing, real_facts))
    preprocessed_fake_facts = list(map(first_preprocessing, fake_facts))

    real_bigrams = list(map(lambda fact: list(ngrams(fact, 2)), preprocessed_real_facts))
    real_trigrams = list(map(lambda fact: list(ngrams(fact, 3)), preprocessed_real_facts))
    fake_bigrams = list(map(lambda fact: list(ngrams(fact, 2)), preprocessed_fake_facts))
    fake_trigrams = list(map(lambda fact: list(ngrams(fact, 3)), preprocessed_fake_facts))
    all_ngrams = real_bigrams + fake_bigrams + real_trigrams + fake_trigrams

    _, combined_frequent_ngrams = count_ngram_frequency(all_ngrams, frequency_threshold=1)

    preprocessed_real_facts = list(map(second_preprocessing, preprocessed_real_facts))
    preprocessed_fake_facts = list(map(second_preprocessing, preprocessed_fake_facts))

    feature_vector = set(combined_frequent_ngrams)

    for word in (preprocessed_real_facts + preprocessed_fake_facts):
        feature_vector.update(word)

    X_real = create_feature_matrix(preprocessed_real_facts, feature_vector, BINARY_FEATURES)
    X_fake = create_feature_matrix(preprocessed_fake_facts, feature_vector, BINARY_FEATURES)

    X = third_preprocessing(np.vstack((X_real, X_fake))) # reducing dimensionality
    y = np.vstack((real_labels, fake_labels))

    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    ### TRAIN ###
    results = train_and_select_best_hyperparameters(X_train, y_train, X_valid, y_valid, BINARY_FEATURES, use_cross_validation=True)

    ### TEST ###
    test_accuracies = test_best_models(results, X_test, y_test)