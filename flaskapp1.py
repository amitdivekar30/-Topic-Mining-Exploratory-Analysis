# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:52:41 2020

@author: aad
"""

#flaskapp.py
from flask import Flask, render_template
from flask import send_file, make_response
from flask import request, jsonify
import csv, re, collections
from collections import Counter
import pandas as pd
import numpy as np
from plot import do_plot,bar_plot
import spacy 
import gensim 
from gensim import corpora


df_visitor= pd.read_csv('E:/Excel_R_Data Science/P26/P26/Chat Transcripts for Project/visitor_chats.csv')
df_v= df_visitor.iloc[:, [1]]
df_v.columns=["content"]

# removing duplicate data
df_v.drop_duplicates(subset='content', keep='first', inplace=True)
# Joinining all the reviews into single paragraph 
text = " ".join(df_v.content)


# Cleaning the texts
import re
import nltk

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
corpus.to_csv('visitor_corpus.csv', index='false')
 
#corpus= pd.read_csv('visitor_corpus.csv')
corpus= corpus.dropna()
corpus.describe
# Joinining all the reviews into single paragraph 
corpus_string = " ".join(corpus.iloc[:,0])
#all word tokens
words_tokens= word_tokenize(corpus_string)
corpus_words = corpus_string.split(" ")



app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Countries', methods=['GET', 'POST'])
def Countries():
    bytes_obj, bytes_obj1, bytes_obj2 = do_plot()
        
    return send_file(bytes_obj, attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/Cities', methods=['GET', 'POST'])
def Cities():
    bytes_obj, bytes_obj1, bytes_obj2 = do_plot()
        
    return send_file(bytes_obj2, attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/Regions', methods=['GET', 'POST'])
def Regions():
    bytes_obj, bytes_obj1, bytes_obj2 = do_plot()
        
    return send_file(bytes_obj1, attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/Frequent_Words_unigram', methods=['GET', 'POST'])
def Frequent_Words_unigram():
    bytes_obj, bytes_obj1, bytes_obj2 = bar_plot()
        
    return send_file(bytes_obj, attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/Frequent_Words_bigram', methods=['GET', 'POST'])
def Frequent_Words_bigram():
    bytes_obj, bytes_obj1, bytes_obj2 = bar_plot()
        
    return send_file(bytes_obj1, attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/Frequent_Words_trigram', methods=['GET', 'POST'])
def Frequent_Words_trigram():
    bytes_obj, bytes_obj1, bytes_obj2 = bar_plot()
        
    return send_file(bytes_obj1, attachment_filename='plot.png',
                     mimetype='image/png')

@app.route('/Wordcloud', methods= ['GET','POST'])
def Wordcloud():       
    from collections import Counter
    word_freqs = Counter(words_tokens) 
    word_freqs = dict(word_freqs)
    word_freqs_js = []
    for key,value in word_freqs.items():
        temp = {"text": key, "size": value}
        word_freqs_js.append(temp)

    max_freq = max(word_freqs.values())
    return render_template('index.html', word_freqs=word_freqs_js, max_freq=max_freq)

@app.route('/Topic_Modelling', methods= ['GET','POST'])
def Topic_Modelling():
    df_text = df_v[['content']]
    df_text['index'] = df_text.index
    #documents = df_text
    
    # function to plot most frequent terms 
    # def freq_words(x, terms = 30): 
    #   all_words = ' '.join([text for text in x]) 
    #   all_words = all_words.split() 
      
    #   fdist = nltk.FreqDist(all_words) 
    #   words_df = pd.DataFrame({'word':list(fdist.keys()),   
    #              'count':list(fdist.values())}) 
    #   # selecting top 20 most frequent words 
    #   d = words_df.nlargest(columns="count", n = terms)      
    #   plt.figure(figsize=(20,5)) 
    #   ax = sns.barplot(data=d, x= "word", y = "count") 
    #   ax.set(ylabel = 'Count') 
    #   plt.show()
      
      
    freq_words(df_text.content)
    
    # # remove unwanted characters, numbers and symbols 
    # df_text['content'] = df_text['content'].str.replace("[^a-zA-Z#]", " ")
    
    
    # from nltk.corpus import stopwords 
    # stop = stopwords.words('english')
    # stop=stop + ['visitor', 'Visitor']
    
    
    # #importing new more stopwords
    # stop_words = []
    # with open("stop.txt") as f:
    #     stop_words = f.read()
    
    # # Convert stopwords to list
    # def Convert(string): 
    #     li = list(string.split("\n")) 
    #     return li
    # s_2=Convert(stop_words)
    
    # #updating list of stopwords and saving into sr_1
    # stop=stop+s_2
    
    # # function to remove stopwords 
    # def remove_stopwords(rev):     
    #   rev_new = " ".join([i for i in rev if i not in stop])      
      # return rev_new 
    # remove short words (length < 3) 
    
    df_text['content'] = df_text['content'].apply(lambda x: ' '.join([w for
                                                w in x.split() if len(w)>2])) 
    # remove stopwords from the text 
    #chats = [remove_stopwords(r.split()) for r in df_text['content']] 
    # make entire text lowercase 
    chats = [r.lower() for r in chats]
    
    
    freq_words(chats, 35)
    
    #!python -m spacy download en #one time run
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner']) 
    def lemmatization(texts, tags=['NOUN','ADJ','VERB']): 
           output = []        
           for sent in texts:              
                 doc = nlp(" ".join(sent))                             
                 output.append([token.lemma_ for token in doc if 
                 token.pos_ in tags])        
           return output
       
        
    tokenized_chats = pd.Series(chats).apply(lambda x: x.split())
    print(tokenized_chats[1])
    
    chats_2 = lemmatization(tokenized_chats)
    print(chats_2[1]) # print lemmatized review
    
    
    chats_3 = []
    for i in range(len(chats_2)):
        #chats_2[i].replace(datum, data)  
        chats_3.append(' '.join(chats_2[i]))
    
    df['chats'] = chats_3
    
    df['chats'] = [item.replace("datum", "data") for item in df['chats']]
    
    freq_words(df['chats'], 35)
    
    chats_2 = pd.Series(chats).apply(lambda x: x.split())
    dictionary = corpora.Dictionary(chats_2)
    
    
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in chats_2]
    # Creating the object for LDA model using gensim library 
    LDA = gensim.models.ldamodel.LdaModel 
    # Build LDA model 
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary,                                     
                    num_topics=10, random_state=100, chunksize=1000,                                     
                    passes=50)
    
    
    topic_list=str(lda_model.print_topics())
    return render_template('index.html', topic_list)


if __name__ == '__main__':
  app.run(debug= True,host="127.0.0.1",port=5000, threaded=True)


