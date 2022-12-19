# Implementation

## Dataset

For dataset visit [kaggle.com](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

## Importing the important libraries

???python
import numpy as np
import pandas as pd
???

## Import Dataset (jupiter or colab)

```python
df = pd.read_csv('spam.csv', encoding='latin-1')
```

## Displaying some data from the dataset
```python
df.sample(5)
```
Here we can see that we have five column .

## Checking the shape of the dataset
```python
df.shape
```

## (1.) DATA CLEANING

## Checking the information of the dataset

```python
df.info()
```

here we can see that in 

` Unnamed: 2  50 non-null     object` 
 `3   Unnamed: 3  12 non-null     object` 
 `4   Unnamed: 4  6 non-null      object`

 have many null values
 So, we have to drop this columns


## Droping the last three columns of the dataset
```python 
df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.head(5)

```
here, `inplace=True` means it will permanently drop the column from the dataset


## Replacing the column name with meaningfull words

```python
df.rename(columns = {'v1' : 'target', 'v2' : 'text'}, inplace=True)
```

## converting the target column data from text to number using label encoder

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
```

What is label encoder?

`converting the labels into a numeric form so as to convert them into the machine-readable form`

What is  fit transform ? 

`Converting the training data from unit to unitless`

## Checking the null values
```python
df.isnull().sum()
```

## Checking dublicate data
```python
df.duplicated().sum()
```

## removing the duplicate data
```python
df.drop_duplicates(keep = 'first')
```

## (2.) EDA (Exploratory Data Analysis)

## counting the spam and ham data present in the database

```python
df['target'].value_counts()
```

## Plotting the graph
```python
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels = ['ham', 'spam'], autopct = '%0.2f')
```
Here, we can see that the `data is imbalanced`

More that `85% of data` is `Ham (Not-Spam)`  and just `15% of data` is `Spam`

## checking the character length of the text data

```python
df['num_characters'] = df['text'].apply(len)
df['num_characters']
```

## checking the number of words of text data

```python
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))
df['num_words']
df.head()
```

## checking the number of sentences of text data

```python
df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df['num_sentences']
df.head()
```

## printing the description of the data

```python
df[['num_characters', 'num_words', 'num_sentences']].describe()
```
Here, in this section we can see the mean, std deviation, min, max, 25%, 50%, 75% of data with respect to ['num_characters', 'num_words', 'num_sentences']

## describing the data on basis of ham

```python
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe()
```

## describing the data on basis of spam

```python
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe()
```

## Checking and printing histogram how much number of charactes are there in ham and in spam data

```python

import seaborn as sns
plt.figure(figsize = (12, 5))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color = 'red')
```

here it will show the clear hist graph of number_of_character present in spam and ham
we can see that number of character count is more in ham and very less number of character count in ham.

## Checking and printing histogram how much number of words are there in ham and in spam data

```python
plt.figure(figsize = (12, 5))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color = 'red')
```

here it will show the clear hist graph of number_of_words present in spam and ham
we can see that words of character count is more in ham and very less number of words count in ham.


## Checking and printing histogramhow much number of sentences are there in ham and in spam data

```python
plt.figure(figsize = (12, 5))
sns.histplot(df[df['target'] == 0]['num_sentences'])
sns.histplot(df[df['target'] == 1]['num_sentences'], color = 'red')
```

here it will show the clear hist graph of number_of_sentence present in spam and ham
we can see that words of sentence count is more in ham and very less number of sentence count in ham.

## Printing the pair plot graph b/w num_sentences, num_words and num_characters

```python
sns.pairplot(df, hue = 'target')
```

## Printing the heat map for checking the corelation

```python
sns.heatmap(df.corr(), annot = True)
```

## (3.) Data Preprocessing

`1.) Lower Case` : Converting all the text into lower case ( helps in the process of preprocessing and in later stages in the NLP application, when you are doing parsing.)

`2.) Tokenization` : the process of  splitting a string, text into a list of tokens.

`3.) Removing Special Character` : @ # $ % etc,.

`4.) Removing stop words` : english stop words eg (is, at, was, etc,.)

`5.) Stemming` : bringing the word at its root form eg : (dancing, danced, dancer = `(dance is the root word )` )

`6.) Punctuation` : !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~


## importing stemming

```python
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('hugging')
```

## importing stop words

```python
from nltk.corpus import stopwords
stopwords.words('english')
```

## importing punctuation strings

```python
import string
string.punctuation
```

## Creating transforming text function

```python

def transform_text(text):
    
    #converting to lower case (eg: (I am Bhavya) ---> (i am bhavya) )
    text = text.lower()
    #creating the tokens for single-single words in the list format (eg:['i', 'am', 'bhavya'])
    text = nltk.word_tokenize(text)
    
    # removing the special character using (is alpha numeric keyword) (eg: (i @ bhavya) ---> (i bhavya))
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # removing english stop words and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #applying stemming (bringing to its root form  eg : dancing, danced, dancer = (dance is the root word ))
    for i in text:
        y.append(ps.stem(i))
    
    # joining the return empty string with y
    return " ".join(y)

```

## Applying the transformed text function to the text column and storing in the newly created column

```python
df['transform_text'] = df['text'].apply(transform_text)
```

## (4.) Creating word cloud of the data

```python
!pip install wordcloud #installing wordcloud
```

```python

# Displaying the higest number of words used in the spam using word cloud
from wordcloud import WordCloud
wc = WordCloud(width = 1500, height = 1500, min_font_size = 20, background_color = 'black')    
spam_wc = wc.generate(df[df['target'] == 1]['transform_text'].str.cat(sep = ' '))
plt.figure(figsize = (12, 6))
plt.imshow(spam_wc)
```

```python
# Displaying the higest number of words used in the ham using word cloud
plt.figure(figsize = (12, 6))
wc = WordCloud(width = 1500, height = 1500, min_font_size = 20, background_color = 'white')
ham_wc = wc.generate(df[df['target'] == 0]['transform_text'].str.cat(sep = ' '))
plt.imshow(ham_wc)
```

`Word Cloud -> Higest number of words used in the message.`


## checking the length and diplaying the top 30 common words used in spam

```python
spam_corpus = []
for msg in df[df['target'] == 1]['transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)
```

```python
from collections import Counter
Counter(spam_corpus).most_common(30)
```

## Creating the tick graph of the most common word in the spam

```python
from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()
```

## checking the length and diplaying the top 30 common words used in ham

```python
ham_corpus = []
for msg in df[df['target'] == 0]['transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
len(ham_corpus)
```

## Creating the tick graph of the most common word in the ham

```python
from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()
```


## (5.) Model Building

# Extracting the features from text using count vectorizer, tfidf vectorizer

```python
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
```
```python
X = tfidf.fit_transform(df['transform_text']).toarray()
X.shape
```

```python
y = df['target'].values
y.shape
```

`CountVectorizer` :  means breaking down a sentence or any text into words by performing preprocessing tasks like converting all words to lowercase, thus removing special characters. tell unique value.

`TF-IDF` : It is used by search engines to better understand the content that is undervalued. For example, when you search for “Coke” on Google, Google may use TF-IDF to figure out if a page titled “COKE” is about: a) Coca-Cola.


## spliting the data into test and train

```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
```


## importing the model libraries
```python
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
```

Easy Explaination [naive bayes and it's types](code.md).


```python
# creating the object Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()
```


```python
# model training code of Gaussian Naive Bayes
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
```

```python
# model training code of Multinomial Naive Bayes
mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
```

```python
# model training code of Bernoulli Naive Bayes
bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
```

```python
# Creating the Pickle file of the model
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
```
