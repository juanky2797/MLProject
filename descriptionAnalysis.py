# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:11:05 2022

@author: BOD
"""

#using NLTK library, we can do lot of text preprocesing
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import re,string,unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data/cleaned_output_with_description.csv',encoding='latin-1', low_memory=False)

nltk.download('stopwords')

tokenizer=ToktokTokenizer()
stopword_list=nltk.corpus.stopwords.words('english')

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
df['DevelopmentDescription']=df['DevelopmentDescription'].apply(denoise_text)

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
df['DevelopmentDescription']=df['DevelopmentDescription'].apply(remove_special_characters)

#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

df['DevelopmentDescription']=df['DevelopmentDescription'].apply(simple_stemmer)

stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

df['DevelopmentDescription']=df['DevelopmentDescription'].apply(remove_stopwords)

norm_desc_train = df.DevelopmentDescription[:40000]
norm_desc_test = df.DevelopmentDescription[20000:]

#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(norm_desc_train)
#transformed test reviews
cv_test_reviews=cv.transform(norm_desc_test)

print('BOW_cv_train:',norm_desc_train.shape)
print('BOW_cv_test:',norm_desc_test.shape)
#vocab=cv.get_feature_names()-toget feature names

#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(norm_desc_train)
#transformed test reviews
tv_test_reviews=tv.transform(norm_desc_test)
print('Tfidf_train:',norm_desc_train.shape)
print('Tfidf_test:',norm_desc_test.shape)

#labeling the sentient data
lb=LabelBinarizer()
#transformed sentiment data
sentiment_data=lb.fit_transform(df['decision'])
print(sentiment_data.shape)

#training the model
mnb=MultinomialNB()
#fitting the svm for bag of words
mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)
print(mnb_bow)
#fitting the svm for tfidf features
mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)
print(mnb_tfidf)
