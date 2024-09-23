def train_and_select_best_hyperparameters(X_train, y_train, X_valid, y_valid, use_cross_validation=False, use_binary_features=False):
    # Combine X_train and X_valid for cross-validation if necessary
    if use_cross_validation:
        X_combined = np.vstack((X_train, X_valid))
        y_combined = np.vstack((y_train, y_valid))

    results = []
    folds = 10

    # Hyperparameter grids for models
    logistic_regression_params = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    ridge_params = {'alpha': [0.1, 1, 10, 100]}
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    perceptron_params = {'penalty': ['l1', 'l2', None], 'alpha': [0.0001, 0.001, 0.01, 0.1]}

    # Logistic Regression (fixed to solver 'liblinear' for simplicity)
    if use_cross_validation:
        log_reg = LogisticRegression(solver='liblinear', max_iter=1000)
        grid_log_reg = GridSearchCV(log_reg, logistic_regression_params, cv=folds)
        grid_log_reg.fit(X_combined, y_combined.ravel())
        best_log_reg_model = grid_log_reg.best_estimator_
        best_log_reg_params = grid_log_reg.best_params_
        valid_acc = grid_log_reg.best_score_
    else:
        best_log_reg_params = None
        best_valid_acc = 0
        for C in logistic_regression_params['C']:
            for penalty in logistic_regression_params['penalty']:
                model = LogisticRegression(C=C, penalty=penalty, solver='liblinear', max_iter=1000)
                model.fit(X_train, y_train.ravel())
                y_pred = model.predict(X_valid)
                valid_acc = accuracy_score(y_valid, y_pred)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_log_reg_model = model
                    best_log_reg_params = {'C': C, 'penalty': penalty}
        valid_acc = best_valid_acc
    results.append((best_log_reg_model, best_log_reg_params, valid_acc))

    # Bernoulli Naive Bayes (for binary features) or Gaussian Naive Bayes
    model_nb = BernoulliNB() if use_binary_features else GaussianNB()
    model_nb.fit(X_train, y_train.ravel())
    y_pred_nb = model_nb.predict(X_valid)
    valid_acc_nb = accuracy_score(y_valid, y_pred_nb)
    results.append((model_nb, None, valid_acc_nb))

    # Ridge Classifier
    if use_cross_validation:
        ridge = RidgeClassifier()
        grid_ridge = GridSearchCV(ridge, ridge_params, cv=folds)
        grid_ridge.fit(X_combined, y_combined.ravel())
        best_ridge_model = grid_ridge.best_estimator_
        best_ridge_params = grid_ridge.best_params_
        valid_acc = grid_ridge.best_score_
    else:
        best_ridge_params = None
        best_valid_acc = 0
        for alpha in ridge_params['alpha']:
            model = RidgeClassifier(alpha=alpha)
            model.fit(X_train, y_train.ravel())
            y_pred = model.predict(X_valid)
            valid_acc = accuracy_score(y_valid, y_pred)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_ridge_model = model
                best_ridge_params = {'alpha': alpha}
        valid_acc = best_valid_acc
    results.append((best_ridge_model, best_ridge_params, valid_acc))

    # Perceptron
    if use_cross_validation:
        perceptron = Perceptron(max_iter=1000)
        grid_perceptron = GridSearchCV(perceptron, perceptron_params, cv=folds)
        grid_perceptron.fit(X_combined, y_combined.ravel())
        best_perceptron_model = grid_perceptron.best_estimator_
        best_perceptron_params = grid_perceptron.best_params_
        valid_acc = grid_perceptron.best_score_
    else:
        best_perceptron_params = None
        best_valid_acc = 0
        for penalty in perceptron_params['penalty']:
            for alpha in perceptron_params['alpha']:
                model = Perceptron(penalty=penalty, alpha=alpha, max_iter=1000)
                model.fit(X_train, y_train.ravel())
                y_pred = model.predict(X_valid)
                valid_acc = accuracy_score(y_valid, y_pred)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_perceptron_model = model
                    best_perceptron_params = {'penalty': penalty, 'alpha': alpha}
        valid_acc = best_valid_acc
    results.append((best_perceptron_model, best_perceptron_params, valid_acc))

    # SVM
    if use_cross_validation:
        svm = SVC()
        grid_svm = GridSearchCV(svm, svm_params, cv=folds)
        grid_svm.fit(X_combined, y_combined.ravel())
        best_svm_model = grid_svm.best_estimator_
        best_svm_params = grid_svm.best_params_
        valid_acc = grid_svm.best_score_
    else:
        best_svm_params = None
        best_valid_acc = 0
        for C in svm_params['C']:
            for kernel in svm_params['kernel']:
                model = SVC(C=C, kernel=kernel)
                model.fit(X_train, y_train.ravel())
                y_pred = model.predict(X_valid)
                valid_acc = accuracy_score(y_valid, y_pred)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    best_svm_model = model
                    best_svm_params = {'C': C, 'kernel': kernel}
        valid_acc = best_valid_acc
    results.append((best_svm_model, best_svm_params, valid_acc))

    return results