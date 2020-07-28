# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:31:50 2020

@author: aad
"""

import nltk 
nltk.download('stopwords') # run this one time
import pandas as pd 
pd.set_option("display.max_colwidth", 200) 

import numpy as np 
import re 
import spacy 
import gensim 
from gensim import corpora 
# libraries for visualization 
import pyLDAvis 
import pyLDAvis.gensim 
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline

df = pd.read_csv('visitor_chats.csv')
df= df.iloc[:, [1]]
df.columns=["content"]
df_text = df[['content']]
df_text['index'] = df_text.index
#documents = df_text

# function to plot most frequent terms 
def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()),   
             'count':list(fdist.values())}) 
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms)      
  plt.figure(figsize=(20,5)) 
  ax = sns.barplot(data=d, x= "word", y = "count") 
  ax.set(ylabel = 'Count') 
  plt.show()
  
  
freq_words(df_text.content)

# remove unwanted characters, numbers and symbols 
df_text['content'] = df_text['content'].str.replace("[^a-zA-Z#]", " ")


from nltk.corpus import stopwords 
stop = stopwords.words('english')
stop=stop + ['visitor', 'Visitor']


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

# function to remove stopwords 
def remove_stopwords(rev):     
  rev_new = " ".join([i for i in rev if i not in stop])      
  return rev_new 
# remove short words (length < 3) 

df_text['content'] = df_text['content'].apply(lambda x: ' '.join([w for
                                            w in x.split() if len(w)>2])) 
# remove stopwords from the text 
chats = [remove_stopwords(r.split()) for r in df_text['content']] 
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


lda_model.print_topics()

# Visualize the topics 
pyLDAvis.enable_notebook() 
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary) 
vis


lda_model.print_topics()


# Joinining all the reviews into single paragraph 
corpus_string = " ".join(df['chats'])
#all word tokens
words_tokens= word_tokenize(corpus_string)

corpus_words = corpus_string.split(" ")

bigram_v = list(nltk.bigrams(words_tokens))
trigram_v = list(nltk.trigrams(words_tokens))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigram_v)
trigram_counts = collections.Counter(trigram_v)

bigram_counts.most_common(20)
trigram_counts.most_common(20)

bigram_df = pd.DataFrame(bigram_counts.most_common(20), columns=['bigram', 'count'])
trigram_df = pd.DataFrame(trigram_counts.most_common(20), columns=['trigram', 'count'])

bigram_df
trigram_df

# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')
d1 = trigram_df.set_index('trigram').T.to_dict('records')

# Create network plot 
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

#G.add_node("china", weight=100)


fig, ax = plt.subplots(figsize=(20, 16))

pos = nx.spring_layout(G, k=2)

# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels = False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=12)
    
plt.show()

#for trigram
# Create network plot 
G = nx.Graph()

# Create connections between nodes
for k, v in d1[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

#G.add_node("china", weight=100)


fig, ax = plt.subplots(figsize=(20, 16))

pos = nx.spring_layout(G, k=2)

# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels = False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0]+.135, value[1]+.045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=12)
    
plt.show()











import pandas as pd

data = pd.read_csv('visitor_chats.csv')
data= data.iloc[:, [1]]
data.columns=["content"]
data_text = data[['content']]
data_text['index'] = data_text.index
documents = data_text

len(documents)
documents[:5]

# Data Preprocessing
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)


import nltk
nltk.download('wordnet')

#Lemmatize example
print(WordNetLemmatizer().lemmatize('went', pos='v'))

#Stemmer Example
stemmer = SnowballStemmer('english')
original_words = ['caresses', 'flies', 'dies', 'mules', 'denied','died', 'agreed', 'owned', 
           'humbled', 'sized','meeting', 'stating', 'siezing', 'itemization','sensational', 
           'traditional', 'reference', 'colonizer','plotted']
singles = [stemmer.stem(plural) for plural in original_words]
pd.DataFrame(data = {'original word': original_words, 'stemmed': singles})

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

doc_sample = documents[documents['index'] == 4310].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

processed_docs = documents['content'].map(preprocess)

processed_docs[:10]

# Bag of words

dictionary = gensim.corpora.Dictionary(processed_docs)

############################################################

############################################################