import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.datasets import load_files
from bs4 import BeautifulSoup
import spacy
import json


train_data = load_files('aclImdb/train', categories=['pos', 'neg'], encoding='utf-8')
reviews = train_data.data
sentiment = train_data.target
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered)

cleaned_docs = [remove_stopwords(doc) for doc in reviews]

print('\nBefore remove stopwords')
print(reviews[0])
print('\nAfter remove stopwords')
print(cleaned_docs[0])

def stemming(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

stemmed = [stemming(doc) for doc in cleaned_docs]
print('\nWith stemming')
print(stemmed[0])

nlp = spacy.load("en_core_web_sm")
def lemmatization(text):
    doc = nlp(text)
    return ' '.join([
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    ])

lemmatized = [lemmatization(doc) for doc in cleaned_docs]
print('\nWith lemmatization')
print(lemmatized[0])

with open('lemmatization.json', 'w', encoding='utf-8') as f:
    content = json.dump(lemmatized, f, ensure_ascii=False)

with open('stemming.json', 'w', encoding='utf-8') as f:
    content = json.dump(stemmed, f, ensure_ascii=False)
