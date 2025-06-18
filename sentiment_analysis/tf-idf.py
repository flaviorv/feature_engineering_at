from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
import pandas as pd
import json

reviews = ''
with open('lemmatization.json', 'r') as f:
    reviews = json.load(f)

train_data = load_files('aclImdb/train', categories=['pos', 'neg'], encoding='utf-8')

tfidf = TfidfVectorizer(token_pattern=r'\b\w{3,}\b', min_df=5, max_df=0.9)
x_train = tfidf.fit_transform(reviews)
y_train = train_data.target

print('TF-IDF scores of document 1')
print(x_train[0].data)