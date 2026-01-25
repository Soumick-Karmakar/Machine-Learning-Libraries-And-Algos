# Imports
import nltk
from nltk.corpus import stopwords

# Corpus - Paragraph
paragraph = """
Narendra Damodardas Modi (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindutva paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress. 

Modi was born and raised in Vadnagar, Bombay State (present-day Gujarat), where he completed his secondary education. He was introduced to the RSS at the age of eight, becoming a full-time worker for the organisation in Gujarat in 1971. The RSS assigned him to the BJP in 1985, and he rose through the party hierarchy, becoming general secretary in 1998. In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat violence and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim. A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him, causing widespread anger and disbelief among the countrys Muslim communities. While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.
"""

# Tokenization - Converts paragraph-sentences-words
resources = ["punkt", "punkt_tab", "wordnet", "stopwords"]
for r in resources:
    nltk.download(r)
sentenses = nltk.sent_tokenize(paragraph) # This package in nltk converts paragraphs into sentenses.

# Cleaning the texts
import re
corpus = []
for i in range(len(sentenses)):
    review = re.sub('[^a-zA-Z]', ' ', sentenses[i])
    review = review.lower()
    corpus.append(review)

# Stemming - Reducing words to their root word
from nltk.stem import PorterStemmer # Needed for stemming (to get the root word)
stemmer = PorterStemmer()
stemmed_words = []
for i in sentenses:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            stemmed_words.append(stemmer.stem(word))


## Lemmatization - Reducing words to their root words
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
for i in corpus:
    words = nltk.word_tokenize(i)
    for word in words:
        if word not in set(stopwords.words('english')):
            print(lemmatizer.lemmatize(word))


## Applying stopwords. Lemmatize
# Cleaning the texts
import re
for i in range(len(sentenses)):
    review = re.sub('[^a-zA-Z]', ' ', sentenses[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)


# Bag Of Words [BOW] Model - Creating matrix of features
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(binary=True, ngram_range=(3,3)) # ngram_range=(1,3) for unigrams, bigrams and trigrams
X = cv.fit_transform(corpus)
print(cv.vocabulary_)
print("================================")
print(corpus[0])
print("================================")
print(X[0].toarray())


# TF-IDF Model - Creating matrix of features
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(ngram_range=(1,3), max_features=3)  # in max_features we can set how many features we want to keep
X = cv.fit_transform(corpus)
print(cv.vocabulary_)
print("================================")
print(corpus[0])
print("================================")
print(X[0].toarray())


## Word2Vec
import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

wv = api.load('word2vec-google-news-300')  # Pre-trained model
print("Vector for king: ",wv['king'])
print("Vector for man: ",wv['man'])
print("Vectors similar to king: ",wv.most_similar('king'))
print("Vectors similar to man: ",wv.most_similar('man'))
print("Similarity between king and queen: ",wv.similarity('king', 'queen'))
print("Similarity between man and woman: ",wv.similarity('man', 'woman'))

vec = wv['king'] - wv['man'] + wv['woman']
print("Vector similar to: ",wv.most_similar([vec]))


### Word2Vec and AvgWord2Vec
import pandas as pd
messages = pd.read_csv("spam_classification.csv", encoding='latin-1', names=['label', 'message'])

# Data Cleaning and Preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

ps = PorterStemmer()

corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

def print_accuracy_and_classification_report(X, y):
    # Train Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    from sklearn.naive_bayes import MultinomialNB
    spam_detect_model = MultinomialNB().fit(X_train, y_train)

    # predictions
    y_pred = spam_detect_model.predict(X_test)

    # scores
    from sklearn.metrics import accuracy_score, classification_report
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))


##### Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500, binary=True, ngram_range=(2,2))
X = cv.fit_transform(corpus).toarray()

print_accuracy_and_classification_report(X, y)


##### Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1,2))
X = tfidf.fit_transform(corpus).toarray()

print_accuracy_and_classification_report(X, y)


##### Creating a simple Word2Vec model
import gensim
model = gensim.models.Word2Vec(corpus, window=5, min_count=2, vector_size=100)
# vector_size is the number of dimensions of the word vectors
print("Vocabulary: ", model.wv.index_to_key)
print("Corpus Count: ", model.corpus_count)
print("Number of epochs: ", model.epochs)
print("Words similar to 'prize': ", model.wv.most_similar('prize'))

##### Creating an Avg Word2Vec model
import numpy as np
def avg_word2vec(doc):
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key], axis=0)  