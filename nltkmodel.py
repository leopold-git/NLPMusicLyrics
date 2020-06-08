import pandas as pd
import nltk
import random
import os
import re
from nltk.tokenize import word_tokenize
import collections
from nltk.metrics import (precision, recall, f_measure)
from sklearn.linear_model import LogisticRegression,SGDClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
import numpy as np
import math


data = pd.read_csv("billboard_lyrics_1964-2015.csv",  encoding='latin-1')


# predict rank given lyrics
# analyze sentiment later
# affective lexicon


#print(data.columns)
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

labeledLyrics['newlabel'] = labels
data['newlabel'] = labels

data['Lyrics'].fillna("FILLER",  inplace = True)

hams = data.loc[data['newlabel'] == 'top 50']
spams = data.loc[data['newlabel']=='bottom 50']
#print(hams['Lyrics'])
#print(spams)

labeledData = []

hams = hams['Lyrics'].tolist()
#print(hams[4])
AllWords = []
text = ""
for i in hams:
    labeledData.append((i, 'top 50'))
    text+= str(i);
for i in spams['Lyrics']:
    labeledData.append((i, 'bottom 50'))
    text += str(i);
random.shuffle(labeledData)
#print(labeledData[0])
# take text containing all words and tokenize it
tokens =  nltk.word_tokenize(text)
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



# get frequency dist of all words
all_words = nltk.FreqDist(words)
#print(all_words)
#take 2000 most prominent words
word_features = list(all_words)[:3000]
print((word_features))

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
    cleaned_row = " ".join(words)
    return cleaned_row

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


def document_features(document):
    document_words = set(document)
    split = song_chunks(document)
    score = findAvgSentimentScore(split)
    #print(document_words)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
        features['emotional score'] = score;
    return features

# need to open the files and read at every labeled review
#featuresets = [(document_features(d), c) for (d,c) in labeledReviews]
# loop and append to featuresets
featuresets = []
for (d,c) in labeledData:
   # print(d,c)
    #docToken = nltk.word_tokenize(d)
    #print(d)
    docToken = clean(d)
    docToken = nltk.word_tokenize(d)
    #print(docToken)
    feat = document_features(docToken)
    featuresets.append((feat, c))
  ## problem


#train_set, test_set = featuresets[400:], featuresets[:400]

### cross val split occurs here  -- 10 folds
num_folds = 10
subset_size = int(round(len(featuresets)/num_folds))

# for the Bayes model
foldAccuracies = []
foldNegativePrecisions = []
foldNegativeRecalls = []
foldNegativeFScores = []
foldPositivePrecisions = []
foldPositiveRecalls = []
foldPositiveFScores = []

for i in range(num_folds):
    cv_test = featuresets[i*subset_size:][:subset_size]
    cv_train = featuresets[:i*subset_size] + featuresets[(i+1)*subset_size:]
    # use NB classifier
    classifier = nltk.NaiveBayesClassifier.train(cv_train)
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    #lg = LogisticRegression_classifier.train(cv_train)

    print('  ')
    print('FOLD ' + str(i))
    print('For this fold:')
    #print('Accuracy on Fold Test Set: ' + str(nltk.classify.accuracy(LogisticRegression_classifier, cv_test)))
    print('Accuracy on Fold Test Set: ' + str(nltk.classify.accuracy(classifier, cv_test)))
    foldAccuracies.append(str(nltk.classify.accuracy(classifier, cv_test)));
    # most informative feauures
    # now get fold stats such as precison, recall, f score
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(cv_test):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    foldPositivePrecisions.append(str(precision(refsets['bottom 50'], testsets['bottom 50'])))
    foldPositiveRecalls.append(str(recall(refsets['bottom 50'], testsets['bottom 50'])))
    foldPositiveFScores.append(str(f_measure(refsets['bottom 50'], testsets['bottom 50'])))
    foldNegativePrecisions.append(str(precision(refsets['top 50'], testsets['top 50'])))
    foldNegativeRecalls.append(str(recall(refsets['top 50'], testsets['top 50'])))
    foldNegativeFScores.append(str(f_measure(refsets['top 50'], testsets['top 50'])))

    print('Positive Precision:', precision(refsets['bottom 50'], testsets['bottom 50']))
    print('Positive Recall:', recall(refsets['bottom 50'], testsets['bottom 50']))
    print('Positive F1-Score:', f_measure(refsets['bottom 50'], testsets['bottom 50']))
    print('Negative Precision:', precision(refsets['top 50'], testsets['top 50']))
    print('Negative Recall:', recall(refsets['top 50'], testsets['top 50']))
    print('Negative F1-Score:', f_measure(refsets['top 50'], testsets['top 50']))
    #classifier.show_most_informative_features(10)

total = 0
totalPrecPos = 0
totalRecallPos = 0
totalFScorePos = 0
totalPrecNeg = 0
totalRecallNeg = 0
totalFScoreNeg = 0
for i in range(0, len(foldAccuracies)):
    total = total + float(foldAccuracies[i])
    totalPrecPos = totalPrecPos + float(foldPositivePrecisions[i])
    totalRecallPos = totalRecallPos + float(foldPositiveRecalls[i])
    totalFScorePos = totalFScorePos + float(foldPositiveFScores[i])
    totalPrecNeg = totalPrecNeg + float(foldNegativePrecisions[i])
    totalRecallNeg = totalRecallNeg + float(foldNegativeRecalls[i])
    totalFScoreNeg = totalFScoreNeg + float(foldNegativeFScores[i])

total_accuracy = total/num_folds
total_pos_prec = totalPrecPos/num_folds
total_pos_recall = totalRecallPos/num_folds
total_pos_fscore = totalFScorePos/num_folds
total_neg_precision = totalPrecNeg/num_folds
total_neg_recall = totalRecallNeg/num_folds
total_neg_fscore = totalFScoreNeg/num_folds
print('---------')
print('Averaged model performance over 10 folds: ')
print('   ')
print('Average accuracy over 10 folds: ' + str(total_accuracy))
print('Average precision for positive class: ' + str(total_pos_prec))
print('Average recall for positive class ' + str(total_pos_recall))
print('Average F-score for positive class ' + str(total_pos_fscore))
print('  ')
print('Average precision for negative class ' + str(total_neg_precision))
print('Average recall for negative class ' + str(total_neg_recall))
print('Average F-score for negative class ' + str(total_neg_fscore))













# take text containing all words and tokenize it















