import pandas as pd
import nltk
import random
import os
import re

from nltk import tokenize
from nltk.tokenize import word_tokenize
import collections
from nltk.metrics import (precision, recall, f_measure)
from sklearn.linear_model import LogisticRegression,SGDClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import wordcloud, WordCloud

analyser = SentimentIntensityAnalyzer()
import numpy as np
import math


data = pd.read_csv("billboard_lyrics_1964-2015.csv",  encoding='latin-1')


# predict rank given lyrics
# analyze sentiment later
# affective lexicon


print(data.columns)
labeledLyrics = data[['Lyrics', 'Rank']].copy()

text = ""
labeledData = []
labels = []

# relabel the rank column to create our own discrete labels
for i in range(len(labeledLyrics)) :
    if pd.isna(labeledLyrics.loc[i, "Lyrics"]):
        labeledLyrics.loc[i, "Lyrics"] = 'FILLER'


    if labeledLyrics.loc[i, "Rank"] < 50:
        labeledLyrics.loc[i, "Rank"] = 'top 50'
    else:
        labeledLyrics.loc[i, "Rank"] = 'bottom 50'
    labeledData.append((labeledLyrics.loc[i, "Lyrics"],labeledLyrics.loc[i, "Rank"]))
    labels.append(labeledLyrics.loc[i, "Rank"])
    text += str(labeledLyrics.loc[i, "Lyrics"])
data['Lyrics'].fillna("FILLER",  inplace = True)

def clean(txt):
    tokens = nltk.word_tokenize(txt)
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    verbs = ['dont', 'filler', 'oh', 'im','make', 'gon', 'let', 'gon na']
    words = [w for w in words if not w in verbs]
    cleaned_row = " ".join(words)
    return cleaned_row

cleaner = lambda x: clean(x)
data['Lyrics'] = data['Lyrics'].apply(cleaner)




# segment different decades
sixties = data.iloc[0:1000]
seventies = data.iloc[1000:2000]
eighties = data.iloc[2000:3000]
nineties = data.iloc[3000:4000]
twothousands = data.iloc[4000:5000]


string60 = sixties.Lyrics.str.cat(sep=',')
string70 = seventies.Lyrics.str.cat(sep=',')
string80 = eighties.Lyrics.str.cat(sep=',')
string90 = nineties.Lyrics.str.cat(sep=',')
string00 = twothousands.Lyrics.str.cat(sep=',')
decades = [string60, string70, string80, string90, string00]
c = sixties['Lyrics']
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
sf = cvec.fit_transform(c)
print('Highest TF-IDF for 1960s: ')
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
print(weights_df.sort_values(by='weight', ascending=False).head(15))
c = eighties['Lyrics']
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
sf = cvec.fit_transform(c)
print('')
print('Highest TF-IDF for 1984-1995: ')
print('')

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
print(weights_df.sort_values(by='weight', ascending=False).head(15))


print('')
print('Highest TF-IDF for 2004-2015: ')
print('')


c = twothousands['Lyrics']
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
sf = cvec.fit_transform(c)

transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(sf)
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
print(weights_df.sort_values(by='weight', ascending=False).head(15))

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200, background_color='white').generate(string80)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()