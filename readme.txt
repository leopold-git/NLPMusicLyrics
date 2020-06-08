README

Jacob Farner, Leopold Ringmayr

Our data is stored in CSV format. While it was originally sourced from Kaggle.com, the dataset has since been made private for unknown	reasons.
It is imported at the beginning of each code file through a direct call to the file, so it works as long as the file is in the project folder, regardless of the file path.

Note: There are external packages that need to be installed and imported for this project, including seaborn, textblob, SentimentIntensityAnalyzer from vaderSentiment.vaderSentiment,
matplotlib, sklearn,  WordNetLemmatizer

LINK TO VIDEO PRESENTATION:
https://www.youtube.com/watch?v=akYpUf3Lh9M


Different components of the project are stored in different files within the PyCharm project: Run each of the files individually for each component.

nltkmodel.py
This file contains the Naive Bayes model with ten fold cross validation, relying on sentiment scores.

sklearnmodels.py
This code will output figures for SVM, Decision Tree, Logistic Regression, and Bayes based off of TF-IDF feature selection.

mostimportantwords.py
This file divides the 50 year dataset into the respective decades and runs TF-IDF analyses to visualize most prominent words and to show terms with highest TF-IDF in different decades

emergingthemes.py
This model searches through lyrics over time using a bag of words model and lexicons we made for religious and sexual references. It then plots them over time to show trends in lyrical content over 50 years.

COMP331_Farner_Ringmayr_Final.pdf
Final Paper fpr this project.