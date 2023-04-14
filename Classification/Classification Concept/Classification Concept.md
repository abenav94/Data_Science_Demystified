## Classification Concept

## What is Classification?

Classification is defined as the process of recognition, understanding, and grouping of objects and ideas into preset categories a.k.a “sub-populations.” With the help of these pre-categorized training datasets, classification in machine learning programs leverage a wide range of algorithms to classify future datasets into respective and relevant categories.

Classification algorithms used in machine learning utilize input training data for the purpose of predicting the likelihood or probability that the data that follows will fall into one of the predetermined categories. One of the most common applications of classification is for filtering emails into “spam” or “non-spam”, as used by today’s top email service providers.

In short, classification is a form of “pattern recognition,”. Here, classification algorithms applied to the training data find the same pattern (similar number sequences, words or sentiments, and the like) in future data sets.

In the below example, the classification model classifies apple and melon from the given set of input data 

## What is Classification Algorithm?

Based on training data, the Classification algorithm is a Supervised Learning technique used to categorize new observations. In classification, a program uses the dataset or observations provided to learn how to categorize new observations into various classes or groups. For instance, 0 or 1, red or blue, yes or no, spam or not spam, etc. Targets, labels, or categories can all be used to describe classes. The Classification algorithm uses labelled input data because it is a supervised learning technique and comprises input and output information. A discrete output function (y) is transferred to an input variable in the classification process (x).

In simple words, classification is a type of pattern recognition in which classification algorithms are performed on training data to discover the same pattern in new data sets.

## Learners in Classification Problems

There are two types of learners.

 1. Lazy Learners: It first stores the training dataset before waiting for the test dataset to arrive. When using a lazy learner, the classification is carried out using the training dataset's most appropriate data. Less time is spent on training, but more time is spent on predictions. Some of the examples are case-based reasoning and the KNN algorithm.

 2. Eager Learners: Before obtaining a test dataset, eager learners build a classification model using a training dataset. They spend more time studying and less time predicting. Some examples are ANN, naive Bayes, and Decision trees.

Now, let us discuss four types of Classification Tasks in Machine Learning.

## Types Of Classification Tasks In Machine Learning
Before diving into the four types of Classification Tasks in Machine Learning, let us first discuss Classification Predictive Modeling.

Classification Predictive Modeling
A classification problem in machine learning is one in which a class label is anticipated for a specific example of input data.

Problems with categorization include the following:

Give an example and indicate whether it is spam or not.

Identify a handwritten character as one of the recognized characters.

Determine whether to label the current user behaviour as churn.

A training dataset with numerous examples of inputs and outputs is necessary for classification from a modelling standpoint.

A model will determine the optimal way to map samples of input data to certain class labels using the training dataset. The training dataset must therefore contain a large number of samples of each class label and be suitably representative of the problem.

When providing class labels to a modeling algorithm, string values like "spam" or "not spam" must first be converted to numeric values. Label encoding, which is frequently used, assigns a distinct integer to every class label, such as "spam" = 0, "no spam," = 1.

There are four different types of Classification Tasks in Machine Learning and they are the following -

1. Binary Classification

2. Multi-Class Classification

3. Multi-Label Classification

4. Imbalanced Classification

Now, let us look at each of them in detail.

Binary Classification:

Those classification jobs with only two class labels are referred to as binary classification.

Examples comprise -

Prediction of conversion (buy or not).

Churn forecast (churn or not).

Detection of spam email (spam or not).

Binary classification problems often require two classes, one representing the normal state and the other representing the aberrant state.

For instance, the normal condition is "not spam," while the abnormal state is "spam." Another illustration is when a task involving a medical test has a normal condition of "cancer not identified" and an abnormal state of "cancer detected."

Class label 0 is given to the class in the normal state, whereas class label 1 is given to the class in the abnormal condition.

A model that forecasts a Bernoulli probability distribution for each case is frequently used to represent a binary classification task.

The discrete probability distribution known as the Bernoulli distribution deals with a situation where an event has a binary result of either 0 or 1. In terms of classification, this indicates that the model forecasts the likelihood that an example would fall within class 1, or the abnormal state.

The following are well-known binary classification algorithms 

1. Logistic Regression, 

2. Support Vector Machines, 

3. Simple Bayes, 

4. Decision Trees

Some algorithms, such as Support Vector Machines and Logistic Regression, were created expressly for binary classification and do not by default support more than two classes.

Multi-Class Classification
Multi-class labels are used in classification tasks referred to as multi-class classification.

Examples comprise -

Categorization of faces.

Classifying plant species.

Character recognition using optical.

The multi-class classification does not have the idea of normal and abnormal outcomes, in contrast to binary classification. Instead, instances are grouped into one of several well-known classes.

In some cases, the number of class labels could be rather high. In a facial recognition system, for instance, a model might predict that a shot belongs to one of thousands or tens of thousands of faces.

Multiclass classification tasks are frequently modeled using a model that forecasts a Multinoulli probability distribution for each example.

An event that has a categorical outcome, such as K in 1, 2, 3,..., K, is covered by the Multinoulli distribution, which is a discrete probability distribution. In terms of classification, this implies that the model forecasts the likelihood that a given example will belong to a certain class label.

For multi-class classification, many binary classification techniques are applicable.

The following well-known algorithms can be used for multi-class classification: 

Progressive Boosting, 

Choice trees, 

Nearest K Neighbors, 

Rough Forest, 

Simple Bayes

Multi-class problems can be solved using algorithms created for binary classification.

In order to do this, a method is known as "one-vs-rest" or "one model for each pair of classes" is used, which includes fitting multiple binary classification models with each class versus all other classes (called one-vs-one).

One-vs-One: For each pair of classes, fit a single binary classification model.

The following binary classification algorithms can apply these multi-class classification techniques:

One-vs-Rest: Fit a single binary classification model for each class versus all other classes.

The following binary classification algorithms can apply these multi-class classification techniques:

Support vector Machine

Logistic Regression

Let us now learn about Multi-Label Classification.

Multi-Label Classification:

Multi-label classification problems are those that feature two or more class labels and allow for the prediction of one or more class labels for each example.

Think about the photo classification example. Here a model can predict the existence of many known things in a photo, such as “person”, “apple”, "bicycle," etc. A particular photo may have multiple objects in the scene.

This greatly contrasts with multi-class classification and binary classification, which anticipate a single class label for each occurrence.

Multi-label classification problems are frequently modeled using a model that forecasts many outcomes, with each outcome being forecast as a Bernoulli probability distribution. In essence, this approach predicts several binary classifications for each example.

It is not possible to directly apply multi-label classification methods used for multi-class or binary classification. The so-called multi-label versions of the algorithms, which are specialized versions of the conventional classification algorithms, include 

Classifier Chains, 

Label Powerset, 

Adapted algorithm (ML-KNN) ,

Multi-label Gradient Boosting, 

Multi-label Random Forests, 

Multi-label Decision Trees

Now, we will look into the Imbalanced Classification Task in detail.

Imbalanced Classification
The term "imbalanced classification" describes classification jobs where the distribution of examples within each class is not equal.

A majority of the training dataset's instances belong to the normal class, while a minority belong to the abnormal class, making imbalanced classification tasks binary classification tasks in general.

Examples comprise -

Clinical diagnostic procedures

Detection of outliers

Fraud investigation

Although they could need unique methods, these issues are modeled as binary classification jobs.

By oversampling the minority class or under sampling the majority class, specialized strategies can be employed to alter the sample composition in the training dataset.

Examples comprise -

SMOTE Oversampling

Random Under sampling

It is possible to utilize specialized modeling techniques, like the cost-sensitive machine learning algorithms, that give the minority class more consideration when fitting the model to the training dataset.

Examples comprise:

Cost-sensitive Support Vector Machines

Cost-sensitive Decision Trees

Cost-sensitive Logistic Regression