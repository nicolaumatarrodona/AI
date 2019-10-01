# Introduction to machine learning with scikit-learn

## Table of Contents

1. What is machine learning, and how does it work? [notebook](01_machine_learning_intro.ipynb)
    - What is machine learning?
    - What are the two main categories of machine learning?
    - What are some examples of machine learning?
    - How does machine learning "work"?

2. Getting started in scikit-learn with the famous iris dataset [notebook](02_getting_started_with_iris.ipynb)
    - What is the famous iris dataset, and how does it relate to machine learning?
    - How do we load the iris dataset into scikit-learn?
    - How do we describe a dataset using machine learning terminology?
    - What are scikit-learn's four key requirements for working with data?

3. Training a machine learning model with scikit-learn [notebook](03_model_training.ipynb)
    - What is the K-nearest neighbors classification model?
    - What are the four steps for model training and prediction in scikit-learn?
    - How can I apply this pattern to other machine learning models?

4. Comparing machine learning models in scikit-learn [notebook](04_model_evaluation.ipynb)
    - How do I choose which model to use for my supervised learning task?
    - How do I choose the best tuning parameters for that model?
    - How do I estimate the likely performance of my model on out-of-sample data?

5. Data science pipeline: pandas, seaborn, scikit-learn [notebook](05_linear_regression.ipynb)
    - How do I use the pandas library to read data into Python?
    - How do I use the seaborn library to visualize data?
    - What is linear regression, and how does it work?
    - How do I train and interpret a linear regression model in scikit-learn?
    - What are some evaluation metrics for regression problems?
    - How do I choose which features to include in my model?

6. Cross-validation for parameter tuning, model selection, and feature selection [notebook](06_cross_validation.ipynb)
    - What is the drawback of using the train/test split procedure for model evaluation?
    - How does K-fold cross-validation overcome this limitation?
    - How can cross-validation be used for selecting tuning parameters, choosing between models, and selecting features?
    - What are some possible improvements to cross-validation?

7. Efficiently searching for optimal tuning parameters [notebook](07_grid_search.ipynb)
    - How can K-fold cross-validation be used to search for an optimal tuning parameter?
    - How can this process be made more efficient?
    - How do you search for multiple tuning parameters at once?
    - What do you do with those tuning parameters before making real predictions?
    - How can the computational expense of this process be reduced?

8. Evaluating a classification model [notebook](08_classification_metrics.ipynb)
    - What is the purpose of model evaluation, and what are some common evaluation procedures?
    - What is the usage of classification accuracy, and what are its limitations?
    - How does a confusion matrix describe the performance of a classifier?
    - What metrics can be computed from a confusion matrix?
    - How can you adjust classifier performance by changing the classification threshold?
    - What is the purpose of an ROC curve?
    - How does Area Under the Curve (AUC) differ from classification accuracy?