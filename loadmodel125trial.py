## for the new text large reviews
# loading the model trial version

#importing the libraries
import nltk
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import NearMiss
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st

#Data collection


earphone= pd.read_csv('Earphones_reviews.csv')

st.title("NLP P125 MODEL")
st.subheader("Input text")
st.write(earphone.text)

messages = earphone.copy()

#Text preprocessing

lematizer = WordNetLemmatizer()
corpus =[]
for i in range(len(earphone.text)):
    review = re.sub('[^a-zA-Z]',' ',str(earphone.text[i]))
    review = review.lower()
    review = review.split()
    review = [lematizer.lemmatize(word) for word in review if word not in (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

earphone['text'] =pd.DataFrame(corpus)

## text to numbers

## Model using tfidf vectorizer
## load the tfidf vectorizer
filename2 = 'tfidf125.sav'
loadtfidf = pickle.load(open(filename2,'rb'))
x= loadtfidf.transform(earphone.text)
x

#loading the model
filename = 'logmodel125.sav'
loadmodel125= pickle.load(open(filename,'rb'))
predictions =loadmodel125.predict(x)
print(predictions)
st.subheader('predictions')
st.write(pd.concat([messages.text,pd.DataFrame(predictions)],axis=1))
predi_probability = loadmodel125.predict_proba(x)
st.subheader('prediction probability')
st.write(predi_probability)

nlp125outputtrial2 = pd.concat([messages.text,pd.DataFrame(predictions)],axis=1)
nlp125outputtrial2.to_excel('nlp125outputtrial2.xlsx')