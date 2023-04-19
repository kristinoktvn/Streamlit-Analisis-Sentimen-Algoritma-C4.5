import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import string
import Sastrawi
import matplotlib.pyplot as plt
import wordcloud
import time


from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, ImageColorGenerator
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS


st.title('Analisis Sentimen Algoritma C4.5')

rad=st.sidebar.radio("Navigation",["Home","Klasifikasi dan Pengujian","Wordcloud"])

#Home Page
if rad=="Home":
    st.text(" ")
    st.text("Berikut ini adalah proses pada Sentimen Analisis Vaksin Booster Algoritma C4.5")
    st.text(" ")
    st.text("1. Klasifikasi dan Pengujian")
    st.text("2. Wordcloud")

#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('indonesian') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

if rad=="Klasifikasi dan Pengujian":
    uploaded_file_pre = st.file_uploader("Choose a file")
    if uploaded_file_pre is not None:
        df = pd.read_csv(uploaded_file_pre)
        st.write(df)
        st.write("")
        kalimat = st.text_area("Enter Text")
        
        if st.button("Check"):
            progress = st.progress(0)
            for i in range (100):
                time.sleep(0.1)
                progress.progress(i+1)

            data_drop_cleaning= df
            #df=load_data()
            

            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            #Perhitungan TF
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(data_drop_cleaning['Hasil'].astype('U'))
            #Perhitungan TF-IDF
            tf = TfidfVectorizer()
            text_tf = tf.fit_transform(data_drop_cleaning['Hasil'].astype('U'))
        
            vectorizer.get_feature_names()

            text_tf.todense()

            df = pd.DataFrame(text_tf.todense().T, 
                            index=vectorizer.get_feature_names(),
                            columns=[f'D{i+1}' for i in range(len(data_drop_cleaning))])

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(text_tf, data_drop_cleaning['Label'], test_size=0.2, random_state=0)
            pos = (y_test == 1).sum()
            neg = (y_test == 0).sum()
            postrain = (y_train == 1).sum()
            negtrain = (y_train == 0).sum()
            total = pos + neg 
            st.write("Testing Data Positive:", pos)
            st.write("Testing Data Negative:",neg)
            st.write("Training Data Positive:", postrain)
            st.write("Training Data Negative:",negtrain)
            data_drop_cleaning['Label'].value_counts()


            classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=1)
            classifier.fit(X_train, y_train)
            

            predictions = classifier.predict(X_test)
            
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
            st.write("")
            st.write("The Confusion Matrix is:")
            st.write(confusion_matrix(y_test,predictions)) 
            st.write("")
            st.write("Recall Score:",recall_score(y_test, predictions).round(2))
            st.write("Precision Score:",precision_score(y_test, predictions).round(2))
            st.write("F1 Score:",f1_score(y_test, predictions).round(2))
            st.write("Accuracy Score:",accuracy_score(y_test, predictions))

            
            st.write("")   
            st.write("")            
            
            st.text("Pengujian Kalimat")
            st.write("Isi teks : ",kalimat)
            transformed_teks = transform_text(kalimat)
            vector = tf.transform([transformed_teks])
            prediksi_label_tree = classifier.predict(vector)[0]
            if prediksi_label_tree == 1:
                st.text("Hasil Pengujian Adalah Positif")
            else :
                st.text("Hasil Pengujian Adalah Negatif")



if rad=="Wordcloud":
    st.set_option('deprecation.showPyplotGlobalUse',False)
    text=st.text_area("Enter text")
    if st.button("Check"):
        progress = st.progress(0)
        for i in range (100):
            time.sleep(0.1)
            progress.progress(i+1)
        w = WordCloud(stopwords=STOPWORDS, width=1600, height=800, max_font_size=200, max_words=80, colormap='Set2', 
        background_color='black').generate("".join(text))
        w.to_file('wordcloud.png')
        st.image('wordcloud.png', use_column_width='auto', caption='Wordcloud',output_format="png")
       
        plt.axis('off')
        st.pyplot()