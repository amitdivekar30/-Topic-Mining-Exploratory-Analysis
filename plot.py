# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 19:47:25 2020

@author: aad
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import re
import nltk
    

def do_plot():
    # Loading
    df = pd.read_csv('E:/Excel_R_Data Science/P26/P26/deployement_bokeh/data/cleaned_structred_data.csv')
    df.dtypes
    df = df.iloc[:, 3: ]
    df.columns
    #country
    countries = list(df['Country_Name'].value_counts().index)
    enquiries = list(df['Country_Name'].value_counts().values)
    fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    plt.bar(countries[:5],enquiries[:5])
    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    # Region
    Region = list(df['Region'].value_counts().index)
    enquiries_region = list(df['Region'].value_counts().values)
    fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    plt.bar(Region[:5],enquiries_region[:5])
  
    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image1 = io.BytesIO()
    plt.savefig(bytes_image1, format='png')
    bytes_image1.seek(0)
    
    # City
    City = list(df['City'].value_counts().index)
    enquiries_city = list(df['City'].value_counts().values)
    fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    plt.bar(City[:5],enquiries_city[:5])
  
    # here is the trick save your figure into a bytes object and you can afterwards expose it via flas
    bytes_image2 = io.BytesIO()
    plt.savefig(bytes_image2, format='png')
    bytes_image2.seek(0)
    
    return bytes_image, bytes_image1, bytes_image2

def bar_plot():
    df_visitor= pd.read_csv('E:/Excel_R_Data Science/P26/P26/Chat Transcripts for Project/visitor_chats.csv')
    df_v= df_visitor.iloc[:, [1]]
    df_v.columns=["content"]
        
    # removing duplicate data
    df_v.drop_duplicates(subset='content', keep='first', inplace=True)    
    # How long are the lenght of the contents
    
    df_v['length'] = df_v['content'].map(lambda text: len(text))       
    #data cleaning
    def preprocessor(text):
        text = re.sub('[^a-zA-Z]', ' ',text)
        text = re.sub('<[^>]*>', '', text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        text = re.sub('[\W]+', ' ', text.lower()) +        ' '.join(emoticons).replace('-', '')
        return text
    
    df_v['content']= df_v["content"].apply(preprocessor)
    
    #tokenizing and lemmatizing
    from nltk.tokenize import word_tokenize
    from nltk.stem.wordnet import WordNetLemmatizer
    porter = nltk.PorterStemmer()    
    lem = WordNetLemmatizer()     
    def tokenizer_lemmatizer(text):
        return[lem.lemmatize(word, "v") for word in text.split()]
        
    from nltk.corpus import stopwords
    stop= stopwords.words('english')
    
    stop=stop + ['visitor']    
    #importing new more stopwords
    stop_words = []
    with open("stop.txt") as f:
        stop_words = f.read()    
    # Convert stopwords to list
    def Convert(string): 
        li = list(string.split("\n")) 
        return li
    s_2=Convert(stop_words)
    
    #updating list of stopwords and saving into sr_1
    stop=stop+s_2
        
    #creating total corpus of mails
    corpus = []
    
    for i in df_v.index.values:
        chat_content=[w for w in tokenizer_lemmatizer(df_v['content'][i]) if w not in stop]
        
        # lem = WordNetLemmatizer()
        # df_v['content'] = [lem.lemmatize(word, "v") for word in df_v['content'] if not word in set(stop)]
        chat_content = ' '.join(chat_content)
        corpus.append(chat_content)
    
           
    corpus= pd.DataFrame(corpus)
    #corpus= pd.read_csv('visitor_corpus.csv')
    corpus= corpus.dropna()
    corpus.describe
    # Joinining all the reviews into single paragraph 
    corpus_string = " ".join(corpus.iloc[:,0])
    #all word tokens
    words_tokens= word_tokenize(corpus_string)
    corpus_words = corpus_string.split(" ")
    
    bigram_v = list(nltk.bigrams(words_tokens))
    trigram_v = list(nltk.trigrams(words_tokens))
    # ### Counter    
    from collections import Counter    
    counter = Counter(words_tokens)
    counter.most_common(20)
    
    counter_bigram = Counter(bigram_v)
    counter_bigram.most_common(50)
    
    counter_trigram = Counter(trigram_v)
    counter_trigram.most_common(50)
    # # Bar Plot
    # =============================================================================
    # convert list of tuples into data frame
    freq_df = pd.DataFrame.from_records(counter.most_common(25), columns =['Word_tokens','Count'])
    
    #Creating a bar plot
    fig = plt.figure()
    freq_df.plot(kind='bar',x='Word_tokens', figsize=(15,10),fontsize=15);
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    
    freq_df = pd.DataFrame.from_records(counter_bigram.most_common(25), columns =['bigram_visitor','Count'])
    
    #Creating a bar plot
    fig = plt.figure()
    freq_df.plot(kind='bar',x='bigram_visitor', figsize=(15,10),fontsize=15);
    bytes_image1 = io.BytesIO()
    plt.savefig(bytes_image1, format='png')
    bytes_image1.seek(0)
    
    freq_df = pd.DataFrame.from_records(counter_trigram.most_common(25), columns =['trigram_visitor','Count'])
    
    #Creating a bar plot
    fig = plt.figure()
    freq_df.plot(kind='bar',x='trigram_visitor', figsize=(15,10),fontsize=15);
    bytes_image2 = io.BytesIO()
    plt.savefig(bytes_image2, format='png')
    bytes_image2.seek(0)
    
    return bytes_image, bytes_image1, bytes_image2
