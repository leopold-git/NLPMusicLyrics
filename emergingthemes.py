import os
import pandas as pd

dataframe = pd.read_csv("billboard_lyrics_1964-2015.csv", encoding='ISO-8859-1')

print(dataframe.describe())

print(dataframe.head())


#looking at occurrences per year
print("Occurrences over years")
print(dataframe.groupby('Year').size().describe())

#print("lyric test")
#print(dataframe['Lyrics'].loc[0])


#need to add column for wordcount
dataframe['Wordcount'] = dataframe['Lyrics'].str.split().str.len()

dataframe['Lyrics'] = dataframe['Lyrics'].str.lower()

#remove missing values in the dataset
dataframe = dataframe.dropna()


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#defining words that trigger a reference - words taken from thesaurus.com
reli_s = 'spirit|god|godly|faith|jesus|lord|heaven|bible|christ|jerusalem|church|holy|soul|heavenly|idol|church|almighty'
sex_s = 'intimate|sensual|make love|sex|body'
reli = dataframe.groupby('Year').apply(lambda x: x['Lyrics'].str.count(reli_s).sum()/x['Wordcount'].sum())
sex = dataframe.groupby('Year').apply(lambda x: x['Lyrics'].str.count(sex_s).sum()/x['Wordcount'].sum())
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(reli, 'y', label = 'Religious References')
ax.plot(sex, 'r', label = 'Sexual References')

ax.set_yticklabels(['{:,.2%}'.format(x) for x in ax.get_yticks()])
legend = ax.legend()


plt.show()