#Saving the model

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
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.layers import LSTM
from keras.layers import Dense
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

#Data collection using amazon review exporter

earphone= pd.read_csv('Reviews_Earphone.csv')
earphone

messages = earphone.copy()

#Text preprocessing

lematizer = WordNetLemmatizer()
corpus =[]
for i in range(len(earphone.Text)):
    review = re.sub('[^a-zA-Z]',' ',earphone.Text[i])
    review = review.lower()
    review = review.split()
    review = [lematizer.lemmatize(word) for word in review if word not in (stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

earphone['Text'] =pd.DataFrame(corpus)

#Forming sentiment column based on the rating given
a1=[]
for i in range(len(earphone.Rating)):
    if earphone.Rating[i]==1:
        a1.append("Negative")
    elif earphone.Rating[i]==2:
        a1.append("Negative")
    elif earphone.Rating[i]==3:
        a1.append("Neutral")
    elif earphone.Rating[i]==4:
        a1.append("Positive")
    else:
        a1.append("Positive")

earphone["sentiment"]=pd.DataFrame(a1)

earphone.sentiment.value_counts()

#Label encoding and balancing the data i.e. doing EDA

label_encoder = LabelEncoder()
earphone['sentiment']=label_encoder.fit_transform(earphone.sentiment)

#sentiment based on rating
earphone.sentiment.value_counts() #2 for positive, 0 for negative and 1 for neutral

## text to numbers

## Model using tfidf vectorizer

tfidf = TfidfVectorizer()
x= tfidf.fit_transform(earphone.Text)
x

y = earphone.sentiment
y

y.value_counts()

### balancing the data

y.value_counts()

## using oversampling method

from imblearn.over_sampling import SMOTE

sm= SMOTE(random_state=15)

xnew4,ynew4=sm.fit_resample(x,y)

ynew4.value_counts()


## Logistic regression model chooosen based on hyperparameter tuning

#train test split
xtrain,xtest,ytrain,ytest = train_test_split(xnew4,ynew4,test_size=0.3,random_state=5)

#Model building
log125= LogisticRegression(penalty='l2',max_iter=5000,C=339.3221771895323)
logmodel125= log125.fit(xtrain,ytrain)
ypredlog125 = logmodel125.predict(xtest)
accuracy_score(ytest,ypredlog125)

#saving the model
filename = 'logmodel125.sav'
pickle.dump(logmodel125,open(filename,'wb'))

#saving the tfidf
filename2 = 'tfidf125.sav'
pickle.dump(tfidf,open(filename2,'wb'))


#loading the model
loadmodel125= pickle.load(open(filename,'rb'))
result = loadmodel125.score(xtest,ytest)
print(result)