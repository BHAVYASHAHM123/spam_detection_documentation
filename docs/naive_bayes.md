
# Naive Bayes

`Naive Bayes` : Suppose we have a dataset of weather conditions and corresponding target variable "Play". So using this dataset we need to decide that whether we should play or not on a particular day according to the weather conditions.

1. Convert the given dataset into frequency tables.

2. Generate Likelihood table by finding the probabilities of given features.

3. Now, use Bayes theorem to calculate the posterior probability.

`Problem`: If the weather is sunny, then the Player should play or not?

`Solution`: To solve this, first consider the below dataset:

![naive bayes](https://github.com/BHAVYASHAHM123/spam_detection/blob/master/spam_docs_image1.png?raw=true)
![naive bayes](https://github.com/BHAVYASHAHM123/spam_detection/blob/master/spam_docs_image2.png?raw=true)
![naive bayes](https://github.com/BHAVYASHAHM123/spam_detection/blob/master/spam_docs_image3.png?raw=true)

## Advantages of Naive Bayes Classifier:

Naive Bayes is one of the fast and easy ML algorithms to predict a class of datasets.
It can be used for Binary as well as Multi-class Classifications.
It performs well in Multi-class predictions as compared to the other Algorithms.
It is the most popular choice for text classification problems.
Disadvantages of Naïve Bayes Classifier:
Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.

## Applications of Naïve Bayes Classifier:

`It is used for Credit Scoring.`

`It is used in medical data classification.`

`It can be used in real-time predictions because Naïve Bayes Classifier is an eager learner.`

`It is used in Text classification such as Spam filtering and Sentiment analysis.`

## Types of Naive Bayes Model:

There are three types of Naive Bayes Model, which are given below:

`Gaussian`: The Gaussian model assumes that features follow a normal distribution. This means if predictors take continuous values instead of discrete, then the model assumes that these values are sampled from the Gaussian distribution.

`Multinomial`: The Multinomial Naïve Bayes classifier is used when the data is multinomial distributed. It is primarily used for document classification problems, it means a particular document belongs to which category such as Sports, Politics, education, etc. The classifier uses the frequency of words for the predictors.


`Bernoulli`: The Bernoulli classifier works similar to the Multinomial classifier, but the predictor variables are the independent Booleans variables. Such as if a particular word is `yes or no` , `true or false` . This model is also famous for document classification tasks.


Content from [javatpoint](https://www.javatpoint.com/machine-learning-naive-bayes-classifier)