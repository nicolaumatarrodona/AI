# AI with Python, R and Javaa


## Data sources

| **Name**                  | **Association Task**  | **Number of Instances**| **Number of Attributes**| **Area** |
| --------------------      |:------------------:| :-----------------:|:-------------------:|:----:|
| [Iris](resources/iris.csv)         | Classification     |        150         |          4          | Life |
| [Titanic](resources/titanic.csv)               | Classification     |        887         |          3           | Life |
| [Diabetes](resources/diabetes.csv)              | Classification     |          768         |          11           | Healthcare |
| [Pima Indians Diabetes](resources/pima_diabetes.csv) | Classification     |         768          |          9           | Healthcare |
| [Breast Cancer Wisconsin](resources/breast_cancer.csv)              | Classification     |          569         |          32           | Healthcare |
| [IBM Watson Telco Chustomer Churn](resources/Telco_Customer_Churn.csv) | Classification     |         7043          |          21           | Retail |


## Python 

**Scikit-learn** - Machine Learning

[`Scikit-learn`](https://scikit-learn.org/stable/) is a machine learning library for Python that features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
 
| Area                             | Algorithm     | Notes   |
| ----------------------------     |:-------------:| -----   |
| **Basic**                         						 |
| [Iris Species](Python/Scikitlearn/Basics/iris.ipynb)              | Random Forest |  It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other. |
| [Titanic: ML from Disaster](/Python/Scikitlearn/Basics/titanic.ipynb)  | Random Forest |  The data consists of demographic and traveling information of 891 of the Titanic passengers, and the goal is to predict whether they survived or not.  |
| **Healthcare**					                         |
| [Diabetes](/Python/Scikitlearn/Healthcare/diabetes.ipynb)  | Random Forest |The dataset contains information about adult Pima females including the number of pregnancies the patient has had, the insulin level the age... |
| [Pima Indians Diabetes](/Python/Scikitlearn/Healthcare/pima_diabetes.ipynb)    | Random Forest |  The datasets consist of several medical predictor (independent) variables and one target (dependent) variable, Outcome. Independent variables include the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.  |
| [Breast Cancer Wisconsin (Diagnostic)](/Python/Scikitlearn/Healthcare/breast_cancer.ipynb)    | Random Forest | Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. |
| **Retail**					                             |
| [Customer Churn](/Python/Scikitlearn/Retail/CustomerChurn/customer_churn_classifier.ipynb)             | Random Forest 		   |  Building a RF classifier model to predict customer churn based on the [IBM Watson Telco Customer Churn dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/), and perform feature selection. |
| [Customer Churn](/Python/Scikitlearn/Retail/CustomerChurn/customer_churn_regressor.ipynb)             | Random Forest 		   |  Building a RF regressor model to predict customer churn based on the [IBM Watson Telco Customer Churn dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/), and perform feature selection. |


**Scikit-learn** - Tutorials

| Tutorials                     | Description           		 | 
| -------------                 |--------------------------------| 	
| [01. Machine Learning Intro](/Python/Scikitlearn/Tutorials/01_machine_learning_intro.ipynb)        | What are the two main categories of machine learning? What are some examples of machine learning? How does machine learning "work"? 				| 
| [02. Getting Startted With Iris](/Python/Scikitlearn/Tutorials/02_getting_started_with_iris.ipynb)    | What is the famous iris dataset, and how does it relate to machine learning? How do we load the iris dataset into scikit-learn? How do we describe a dataset using machine learning terminology? What are scikit-learn's four key requirements for working with data?  |  
| [03. Model Training](/Python/Scikitlearn/Tutorials/03_model_training.ipynb) | What is the K-nearest neighbors classification model? What are the four steps for model training and prediction in scikit-learn? How can I apply this pattern to other machine learning models? |  
| [04. Model Evaluation](/Python/Scikitlearn/Tutorials/04_model_evaluation.ipynb) | How do I choose which model to use for my supervised learning task? How do I choose the best tuning parameters for that model? How do I estimate the likely performance of my model on out-of-sample data? |
| [05. Linear Regression](/Python/Scikitlearn/Tutorials/05_linear_regression.ipynb) | How do I use the pandas library to read data into Python? How do I use the seaborn library to visualize data? What is linear regression, and how does it work? How do I train and interpret a linear regression model in scikit-learn? What are some evaluation metrics for regression problems? How do I choose which features to include in my model? |  
| [06. Cross Validation](/Python/Scikitlearn/Tutorials/06_cross_validation.ipynb) | What is the drawback of using the train/test split procedure for model evaluation? How does K-fold cross-validation overcome this limitation? How can cross-validation be used for selecting tuning parameters, choosing between models, and selecting features? What are some possible improvements to cross-validation? |
| [07. Grid Search](/Python/Scikitlearn/Tutorials/07_grid_search.ipynb) | How can K-fold cross-validation be used to search for an optimal tuning parameter? How can this process be made more efficient? How do you search for multiple tuning parameters at once? What do you do with those tuning parameters before making real predictions? How can the computational expense of this process be reduced? |
| [08. Classification Metrics](/Python/Scikitlearn/Tutorials/08_classification_metrics.ipynb) | What is the purpose of model evaluation, and what are some common evaluation procedures? What is the usage of classification accuracy, and what are its limitations? How does a confusion matrix describe the performance of a classifier? What metrics can be computed from a confusion matrix? How can you adjust classifier performance by changing the classification threshold? What is the purpose of an ROC curve? How does Area Under the Curve (AUC) differ from classification accuracy? |

**Keras** - Deep Learning

[`Keras`](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation. 
 
| Area                             | Algorithm     | Notes   |
| ----------------------------     |:-------------:| -----   |
| **Basic**                         						 |
| [Iris Species]()                 | todo          |  todo   | 
| [Titanic: ML from Disaster]()    | todo          |  todo   |
| **Healthcare**					                         |
| [Diabetes]()                     | todo          |  todo   |
| [Pima Indians Diabetes]()        | todo          |  todo   |
| **Retail**					                             |
| [Customer Churn](/Python/Keras/Retail/CustomerChurn/customer_churn.ipynb) | MLP |  Build an ANN (Deep Learning) to predict customer churn based on the [IBM Watson Telco Customer Churn dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/).|


## R

**Healthcare.ai** - Machine Learning in healthcare

[`healthcare.ai`](https://healthcare.ai/) is a Python and R library designed to streamline healthcare machine learning by including functionality specific to healthcare, as well as simplifying the workflow of creating and deploying models.
 
| Area                             | Algorithm     | Notes   |
| ----------------------------     |:-------------:| -----   |
| [Diabetes]()                     | Random Forest           |  Healthcareai comes with a built in dataset documenting diabetes among adult Pima females. Once you attach the package, the dataset is available in the variable pima_diabetes.    |

**Keras** - Deep Learning

| Area                             | Algorithm     | Notes   |
| ----------------------------     |:-------------:| -----   |
| **Retail**                         						 |
| [Customer Churn]()               | MLP 		   |  Deep Learning to predict customer churn based on the [IBM Watson Telco Customer Churn dataset](https://www.ibm.com/communities/analytics/watson-analytics-blog/predictive-insights-in-the-telco-customer-churn-data-set/).|


## Java

**Weka** - Machine Learning

[`Weka`](https://www.cs.waikato.ac.nz/ml/weka/) is a collection of machine learning algorithms for data mining tasks. It contains tools for data preparation, classification, regression, clustering, association rules mining, and visualization.


# Machine Learning vs Deep Learning 

Machine Learning (ML) is a way to implement artificial intelligence. Similar to AI, machine learning is a branch of computer science in which you devise or study the design of algorithms that can learn. 

There are various machine learning algorithms like:

- Decision trees
- Naive Bayes
- Random forest
- Support vector machine
- K-nearest neighbor
- Gaussian mixture model

For now, understand that in machine learning you use one of the algorithms as mentioned above which provides the computer the ability to automatically learn and understand without being programmed time and again.

Now the question is how will the computer learn automatically?
Well, the answer is data. You feed in the data having different attributes or features that the algorithms have to understand and give you a decision boundary based on the data you provide it. Once the algorithm has learned and interpreted the data, meaning it has trained itself, you can then put your algorithm in the testing phase and without explicitly programming it, input a test data point and expect it to give you some results.

For example: Let's say you have to predict the price of the house, given a dataset comprising of the cost of a home and the number of rooms in the house, and 1000 houses with similar attributes. Both the price and the number of rooms are features. Now your goal is to feed these two features into let's say decision trees algorithm. Your input, in this case, will be the number of rooms, and the algorithm has to predict the price of the house. The algorithm will try to learn a relationship between the number of rooms and cost of the home.
Now, at test time you will give the algorithm let's say three (number of rooms) as input, and it should be able to predict the price of the house accurately!




