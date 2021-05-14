
# Introduction
The experiment aims to implement multinomial Naïve Bayesian algorithms from scratch, and try to modify the algorithm to get better performance.
*Ask my lecturers why we are not allowed to use pandas to manipulate data!!!!*

# Implementation
The project is divided into three parts:
1. Reading data,
2. vectorize the abstracts,
3. train and test model.

## 1. Reading data
For this part, We created a `DataFrame` class, this class is designed and coded to simulate a `pan das.DataFrame`. Therefore the object of this class can be considered a dataframe, with some basic functions enabled. `train_test_split` is implemented as a instance object to split the training and testing set. 
## 2. Count  Vectorizer
We created a CountVectorizer class for this part. `CountVectorizer` is created to simulate some basic functions provided by `sklearn.feature_extraction.CountVectorizer`. Any instance of this class is a vectorizer, use `~.fit` to feed it with texts to generate a **bag of words**, and after that use `~.transform` to transform any abstracts into a **text vector**. 
## 3. MultinomialNB
We created a MultinomialNB class for this part. `MultinomialNB` is created to simulate some basic functions provided by `sklearn.naive_bayes.MultinomialNB`. Any instance of this class is a classifier. `~.fit` function takes train set and train the classifier by calculating parameters like **priors** and **likelihoods**, then store them as some instance variables, TF-IDF can be enabled with passing the boolean parameter `idf`. `~.predict` uses the parameters generated form `~.fit` to calculate the posterior probability. Due to technical issue, metric methods are integrated in MultinomialNB class. 
### 3.1 IDF smoothing
IDF originally is calculated by $$\forall w\in W, IDF = \log(\frac{N}{D_w})$$However in case there are no documents that have a certain word. We made it $$\forall w\in W, IDF = \log(\frac{N+1}{D_w+1})+1$$
### 3.2 Likelihood Smoothing
likelihood originally is calculated by $$\mathbb{P}(w_i|V) = \frac{w_i}{\sum_{x=0}^{x= |V| } x_v}$$However, in case some words have likelihoods of 0 which makes the posterior probability 0, We made it $$\mathbb{P}(w_i|V) = \frac{w_i+1}{\sum_{x=0}^{x= |V| } x_v+|V| }$$

## Experiment Designs
The experiment is implemented in two conditions. In order to make the running time reasonable, 1000 features are selected. First We try the algorithm without TF-IDF, then We try the algorithm with TF-IDF. 

Cross validation is used to evaluate overall performance of the clasifier on whole dataset. With repeatly doing cross validation 16 times on randomly shuffled dataset to make sure **chance is not acting alone**.

## Expected Outcomes
Applying TF-IDF instead of simply using the frequency of each word reflects the importance of a word to a document in a bag of word. For any word, The more documents that this word is in, the lower IDF it is. The less documents this word is in, the higher IDF it is. 

# Results and Findings
Let $H_0$be the mean accuracy of two situations are same.

Let $H_A$be the mean accuracy of two situations are different.

The repeated cross_val sections shows the sample mean accuracy of the classifier without tf-idf is 0.8859531250000001, and the $\sigma$	is 0.001247556204896173, and those of the classifier with tf-idf are 0.8859531250000001	and 0.0012475562048961737. 

The sample size of both samples are 16, therefore degree of freedom is 30. 

With two samples t-test, we can see the p-value = 0.016. This rejects our null hypothesis that the mean accuracy with no TF-IDF is same as that of using TF-IDF. With the mean accuracies returned from the repeated cross_val. With 95% confidence, we can say, the mean accuracy of using TF-IDF is significantly higher than that of not using TF-IDF.
