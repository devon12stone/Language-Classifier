import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#-------------------Input File--------------------------------
# format: "phrase", "language"
input_data = pd.read_csv("lang_data.csv")

# drop rows where "phrase" is blank
input_data = input_data.dropna()

#-------------------Cleaning Data-----------------------------

# numpy represenations of the phrases and languages in the input file
phrases = input_data['text'].values
languages = input_data['language'].values

# possible punctuation marks
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

for phrase in phrases:

    # remove capital letters
    phrase = phrase.lower()

    #remove punctuation
    blank = ""
    for char in phrase:
        if char not in punctuations:
            blank = blank + char
    phrase = blank

    # remove additional whitespace
    phrase = " ".join(phrase.split())

#-------------------Training and Test Data----------------------

# 80:20 training:test split
total_phrases = len(phrases)
percent_80 = int(0.8 * total_phrases)
print(percent_80)

# training sets
train_phrase = phrases[:percent_80]
train_targets = languages[:percent_80]

# test sets
test_phrases = phrases[percent_80:]
test_targets = languages[percent_80:]




#-------------------Tokenizing Phrases-------------------------

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_phrase)


#-------------------Training Classifier-----------------------

clf = MultinomialNB().fit(X_train_counts, train_targets)

# tokenizing the test Data
X_test_counts = count_vect.transform(test_phrases)

# predicting with test Data

predict = clf.predict(X_test_counts)


#-------------------Overall Result------------------------------

# overall predictive accuracy
print("Overall Predicitve Accuracy")
print(round(np.mean(predict == test_targets), 3))
print(" ")

#-------------------Each language Result-------------------------

eng_targets = test_targets[test_targets == "English"]
afr_targets = test_targets[test_targets == "Afrikaans"]
ned_targets = test_targets[test_targets == "Nederlands"]

eng_pred = []
afr_pred = []
ned_pred = []

for tar in range(len(test_targets)):

    if test_targets[tar] == "English" and predict[tar] == "English":
        eng_pred.append(predict[tar])

    elif test_targets[tar] == "Afrikaans" and predict[tar] == "Afrikaans":
        afr_pred.append(predict[tar])

    elif test_targets[tar] == "Nederlands" and predict[tar] == "Nederlands":
         ned_pred.append(predict[tar])

print("Enlgish Predicitve Accuracy")
print(round(len(eng_pred)/len(eng_targets), 3))
print(" ")

print("Afrikaans Predicitve Accuracy")
print(round(len(afr_pred)/len(afr_targets), 3))
print(" ")

print("Ductch Predicitve Accuracy")
print(round(len(ned_pred)/len(ned_targets), 3))
