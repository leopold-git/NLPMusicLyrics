import pandas as pd
import nltk
import random
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob

data = pd.read_csv("billboard_lyrics_1964-2015.csv", encoding='latin-1')


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
        labeledLyrics.loc[i, "Rank"] = 'lower 50'
    labeledData.append((labeledLyrics.loc[i, "Lyrics"],labeledLyrics.loc[i, "Rank"]))
    labels.append(labeledLyrics.loc[i, "Rank"])
    text += str(labeledLyrics.loc[i, "Lyrics"])

labeledLyrics['newlabel'] = labels
data['newlabel'] = labels
#data.apply(lambda col: col.drop_duplicates().reset_index(drop=True))

print('finding sentiments within songs to build model...')
# function used for text cleaning
def clean(txt):
    re.sub("'","", txt)
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
    cleaned_row = [''.join(t) for t in words]
    cleaned_row = str(cleaned_row)
    return cleaned_row


# take text containing all words and tokenize it




def document_features(document):
    document_words = set(document)
    #print(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features
featuresets = []

#data.dropna(inplace=True,axis=1)
data['Lyrics'].fillna("FILLER", inplace = True)

sentiments = []
subj = []
pol = []

cleaner = lambda x: clean(x)
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity
data['cleanedup'] = data['Lyrics'].apply(cleaner)
data['polarity'] = data['cleanedup'].apply(pol)
data['subjectivity'] = data['cleanedup'].apply(pol)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
import numpy as np
import math


# 3 methods below used for assigning an emotion to a song
# chunks a song into 10 equal parts
def song_chunks(lyrics):
    length = len(lyrics)
    if(length < 50):
        split = ['placeholder', 'placeholder','placeholder','placeholder','placeholder','placeholder','placeholder','placeholder','placeholder','placeholder',]
        return split
    #print(lyrics)
    #print(length)
    partition_size = math.floor(length/ 10)
    start = np.arange(0, length, partition_size)
    split = []
    for part in range(10):
        split.append(lyrics[start[part]:start[part]+partition_size])
    return split

# takes in a lyric chunk and calculates the avg sentiment score from all chunks
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score

# uses above method to loop thru all chunks for a song and finds average sentiment
# avg sentiment score is used to label the song with an emotion
def findAvgSentimentScore(split):
    sum = 0
    for i in split:
        x = sentiment_analyzer_scores(i)
        sum+= x['compound']
    final = sum/len(split)
    fin = 0
    if final < 0:
        fin = -1
    elif final > 0 and final < 0.5:
        fin = 0.5
    else:
        fin = 1
    #print(final)
    return fin

# assign sentiment scores to each song based on average score from song lyric chunks
chunked = lambda x: song_chunks(x)
vader_scores = lambda x: findAvgSentimentScore(x)
data['chunked'] = data['Lyrics'].apply(chunked)
data['vader_scores'] = data['chunked'].apply(vader_scores)
test = labeledLyrics.loc[1, 'Lyrics']
print(data['vader_scores'] )
#clean_test = clean(test)
#print(clean_test)


data.sample(frac=1)


X = data['cleanedup']
Y = data['newlabel']
print('')
print('Preparing cleaned data using TF-IDF for features...')
tf = TfidfVectorizer(min_df=0.5, stop_words='english')
# split into test and training
trainingMessages, testMessages, trainingLabels, testLabels = train_test_split(X, Y, test_size=0.2, random_state=4)
trainingMessagesTfIdf = tf.fit_transform(trainingMessages)
#print(trainingMessagesTfIdf.toarray())
#print(trainingMessagesTfIdf)
testMessagesCV = tf.fit_transform(testMessages)
# need to open the files and read at every labeled review
#featuresets = [(document_features(d), c) for (d,c) in labeledReviews]
# loop and append to featuresets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

target_labels = ['top 50', 'lower 50']
print('')
print('Building Logistic Regression Model:')
# now logistic regression
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression()
lg.fit(trainingMessagesTfIdf.todense(),trainingLabels)
lgpred = lg.predict(testMessagesCV.todense())
print(classification_report(testLabels, lgpred, target_names=target_labels))
print('Logistic Regression accuracy Score: ', accuracy_score(testLabels,lgpred))



'''Gaussian Bayes'''
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(trainingMessagesTfIdf.todense(),trainingLabels)
gnb_pred = gnb.predict(testMessagesCV.todense())
print(classification_report(testLabels, gnb_pred, target_names=target_labels))
print('Bayes accuracy Score: ', accuracy_score(testLabels,gnb_pred))


"Decision Tree Classifier"
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(trainingMessagesTfIdf.todense(),trainingLabels)
dtree_pred = dtree_model.predict(testMessagesCV.todense())
print(classification_report(testLabels, dtree_pred, target_names=target_labels))
print('Decision Tree accuracy Score: ', accuracy_score(testLabels,dtree_pred))


'''Support Vector Machine Classifier'''
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(trainingMessagesTfIdf.todense(), trainingLabels)
svm_pred = svm_model_linear.predict(testMessagesCV.todense())

###TRYING TO ACCOUNT FOR DIFFERENT SIZES HERE
print(classification_report(testLabels, svm_pred, target_names=target_labels))
print('SVM accuracy Score: ', accuracy_score(testLabels,svm_pred))

print('Logistic Regression accuracy Score: ', accuracy_score(testLabels,lgpred))
print('Bayes accuracy Score: ', accuracy_score(testLabels,gnb_pred))
print('Decision Tree accuracy Score: ', accuracy_score(testLabels,dtree_pred))
print('SVM accuracy Score: ', accuracy_score(testLabels,svm_pred))