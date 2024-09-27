# Classifying Fake and Real Facts City Facts
## Problem Setup
This experiment investigates a text classification problem using a dataset of real and fake facts created by a generative AI tool. While fake news detection often involves complex NLP techniques, this study focuses on evaluating the effectiveness of simple linear classifiers to answer two questions:

1. Can a linear classifier distinguish between real and fake facts about cities?
2. Does the choice of classifier matter?

Several hyperparameter tuning techniques and feature representations were explored[^1] in this experiment. Through manual testing, it was determined that binary feature representation combined with a static train-validation split yielded the best performance. Therefore, the report focuses on the implementation and results of this approach.

## Dataset Generation and Experimental Procedure
The dataset was created by prompting[^2] the online ChatGPT-4o [1] for 1000 real and 1000 fake facts about 10 different cities[^3] - these were stored in two UTF-8 encoded files. Initial preprocessing involved expanding contractions, removing special characters, cleaning numbers, and tokenizing the text. After removing stop words, the most frequently occuring bigrams and trigrams were retained. Porter stemming and lemmatization were later applied in order to reduce words to their base forms [2]. Given the short nature of the sentences, the BINARY_FEATURES variable was set to True thereby counting features as binary values and guiding the use of the
Bernoulli Naive Bayes [3] as one of the classifiers.

After the feature matrix was converted into a numpy array and zero columns were removed to reduce dimensionality, train_test_split was used to separate the data into training (64%), validation (16%), and test (20%) sets [4] . After selecting the best hyperparameters using a static train-validation approach, models were evaluated for accuracy on the test set.

## Range of Parameter Settings
Given the small dataset[^4] size, regularization (L1, L2) was applied in Logistic Regression [5], Linear SVM [6], and Perceptron [7] classifiers to reduce overfitting. Alpha smoothing was used in Bernoulli Naive Bayes [3] to avoid zero probabilities, and maximum iterations were set to 10,000 to ensure convergence. The hyperparameters of the classifiers referred to in this report include: Logistic Regression: C: [0.01, 0.1, 1, 5], penalty: ['l1', 'l2'], BernoulliNB: alpha: [0.001, 0.01, 0.1, 1.0], Perceptron: penalty: ['l1', 'l2', None], alpha: [0.001, 0.01, 0.1, 1.0], LinearSVC: C: [0.001, 0.01, 0.1], loss: ['hinge', 'squared_hinge'].

## Results and Conclusions
The optimal hyperparameters for each model resulted in a consistent validation accuracy of 95.94%, suggesting a clear distinction between features for real and fake facts. This implies that the models are learning similar patterns, or the validation set contains easily classifiable examples. During testing, BernoulliNB outperformed the other 3 classifiers with 95.00% test accuracy, likely benefiting from binary feature representation. Notably, BernoulliNB’s close alignment between validation and test accuracy indicates it effectively captures meaningful data patterns. The model performances are detailed below.
| Model                | Hyperparameters                        | Validation Accuracy | Test Accuracy |
|----------------------|----------------------------------------|---------------------|---------------|
| LinearSVC            | {'C': 0.1, 'loss': 'squared_hinge'}    | 95.94%              | 94.50%        |
| Perceptron           | {'penalty': 'l1', 'alpha': 0.001}      | 95.94%              | 92.00%        |
| BernoulliNB          | {'alpha': 0.1}                         | 95.94%              | 95.00%        |
| LogisticRegression   | {'C': 1, 'penalty': 'l2'}              | 95.94%              | 94.75%        |

To conclude, this experiment confirmed that linear classifiers can effectively distinguish between real and fake facts, with all models surpassing 90% generalization accuracy. BernoulliNB performed best, likely due to its suitability for binary features, where word presence or absence was highly indicative. Logistic Regression and LinearSVC also showed strong performance, ranking second and third, respectively. The Perceptron classifier lagged slightly, possibly due to the dataset's non-linear characteristics or strong regularization. Overall, the results highlight that the choice of the model influences performance.

## Limitations of the Study
The study faced several limitations, primarily the small dataset size (2,000 samples), which increased the risk of overfitting. Additionally, focusing on only 10 cities reduced dataset diversity, which may affect how well the model can be applied to other contexts. The uniform writing style generated by ChatGPT-4o may have simplified the task, making it easier for models to find patterns. More varied text would have posed a greater challenge. Lastly, restricting the analysis to linear classifiers limits the exploration of complex patterns, and relying solely on accuracy is insufficient without evaluating other metrics like precision, recall, and the confusion matrix.

## References
[1] OpenAI, "ChatGPT: Language Model," 2024. [Online]. Available: https://chat.openai.com. [Accessed: September 2024].

[2] DataCamp, "Stemming and Lemmatization in Python," DataCamp, Feb. 28, 2023. [Online]. Available: https://www.datacamp.com/tutorial/stemming-lemmatization-python. [Accessed: Sep. 21 , 2024].

[3] Scikit-learn, "sklearn.naive_bayes.BernoulliNB," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html. [Accessed: Sep. 19, 2024].

[4] Scikit-learn, "sklearn.model_selection.train_test_split," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html. [Accessed: Sep. 19, 2024].

[5] Scikit-learn, "sklearn.linear_model.LogisticRegression," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. [Accessed: Sep. 19, 2024].

[6] Scikit-learn, "sklearn.svm.LinearSVC," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html. [Accessed: Sep. 13, 2024].
[7] Scikit-learn, "sklearn.linear_model.Perceptron," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html. [Accessed: Sep. 19, 2024].

[8] Scikit-learn, "sklearn.naive_bayes.GaussianNB," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html. [Accessed: Sep. 19, 2024].

[9] Scikit-learn, "sklearn.model_selection.GridSearchCV," Scikit-learn Documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html. [Accessed: Sep. 20, 2024].

[10] Stack Overflow, "What is C parameter in sklearn logistic regression?," Stack Overflow, 2021. [Online]. Available: https://stackoverflow.com/questions/67513075/what-is-c-parameter-in-sklearn-logistic-regression. [Accessed: Sep. 20, 2024].

[11] Graphite Note, "How Much Data is Needed for Machine Learning?". [Online]. Available: https://graphite-note.com/how-much-data-is-needed-for-machine-learning/#:~:text=This%20approach%20involves%20making%20an,have%20at%20least%20100%20rows. [Accessed Sep. 26, 2024].

[^1]: This is why certain functions in the code, such as create_feature_matrix and train_and_select_best_hyperparameters, include conditional statements that allow users to choose between binary features or frequency counts for dataset representation. Similarly, users can select between a static train-validation split for hyperparameter tuning or opt for a more robust k-fold cross validation technique.
[^2]: Prompts used: “Generate 100 real facts about <city-name>” and “Generate 100 fake facts about <city-name>” where <city-name> was replaced by the appropriate city mentioned in the following footnote.
[^3]: The 10 cities included in fakes.txt and facts.txt: New York City, Moscow, Vancouver, Cairo, Los Angeles, Beijing, Tokyo, Buenos Aires, Bangalore, Paris.
[^4]: Having 2,000 samples is insufficient for the 2,689 features. As [11] suggests, a common rule of thumb is to have "at least ten times as many data points as there are features" for reliable model performance.