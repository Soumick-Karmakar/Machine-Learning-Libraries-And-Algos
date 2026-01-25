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

