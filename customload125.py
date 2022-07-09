# loading the custom input version

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


def main():
    st.title("NLP SENTIMENT ANALYSIS")
    st.subheader("Streamlit Project P125")

    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # layout
        col1,col2= st.columns(2)
        if submit_button:

            with col1:
                st.info("Results")
                lematizer = WordNetLemmatizer()
                corpus=[]
                review= re.sub('[^a-zA-Z]',' ',str(raw_text))
                review= review.lower()
                review = review.split()
                review = [lematizer.lemmatize(word) for word in review if word not in (stopwords.words('english'))]
                review = ' '.join(review)
                corpus.append(review)
                
                ## text to numbers

                ## Model using tfidf vectorizer
                ## load the tfidf vectorizer
                filename2 = 'tfidf125.sav'
                loadtfidf = pickle.load(open(filename2,'rb'))
                x= loadtfidf.transform(corpus)

                
                #loading the model
                filename = 'logmodel125.sav'
                loadmodel125= pickle.load(open(filename,'rb'))
                predictions =loadmodel125.predict(x)
                print(predictions)
                st.subheader('predictions')
                
                b= {"Raw Text":raw_text,"sentiment_value":predictions}
                c=pd.DataFrame(b)
                st.write(c)
                
                # Emoji
                if predictions==2:
                    st.markdown("Sentiment:: Positive :smiley: ")
                elif predictions==0:
                    st.markdown("Sentiment:: Negative :angry: ")
                else:
                    st.markdown("Sentiment:: Neutral 😐 ")
                    
                    









    else:
        st.subheader("About")
        st.write("In this project we are giving reviews of product in text format to the model and Model is built in such a way which can Extract and Classify sentiment of the reviews on a product into positive, negative and neutral category.")


if __name__ == '__main__':
    main()
