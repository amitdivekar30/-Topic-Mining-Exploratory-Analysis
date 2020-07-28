# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:48:43 2020

@author: aad
"""

# EDA unstructerd data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df_visitor= pd.read_csv('visitor_chats.csv')
df_v= df_visitor.iloc[:, [1]]
df_v.columns=["content"]
df_v.isnull().sum() #no na values
df_v.shape
df_v.info()
df_v.describe()
df_v.dtypes

# removing duplicate data
df_v.drop_duplicates(subset='content', keep='first', inplace=True)

# How long are the lenght of the contents

df_v['length'] = df_v['content'].map(lambda text: len(text))

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

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus1 = []
for i in range(0, len(df_v['content'])):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', df_v['content'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stop]
    review = ' '.join(review)
    corpus1.append(review)
    
corpus= pd.DataFrame(corpus)
corpus.to_csv('visitor_corpus.csv', index='false')

corpus= pd.read_csv('visitor_corpus.csv')
corpus= corpus.dropna()
corpus.describe

# Joinining all the reviews into single paragraph 
corpus_string = " ".join(corpus.iloc[:, 1])
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
freq_df = pd.DataFrame.from_records(counter.most_common(50), columns =['Word_tokens','Count'])

#Creating a bar plot
freq_df.plot(kind='bar',x='Word_tokens', figsize=(15,10),fontsize=15);

freq_df = pd.DataFrame.from_records(counter_bigram.most_common(50), columns =['bigram_visitor','Count'])

#Creating a bar plot
freq_df.plot(kind='bar',x='bigram_visitor', figsize=(15,10),fontsize=15);

freq_df = pd.DataFrame.from_records(counter_trigram.most_common(50), columns =['trigram_visitor','Count'])

#Creating a bar plot
freq_df.plot(kind='bar',x='trigram_visitor', figsize=(15,10),fontsize=15);


# Simple word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud_tot = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_string)

plt.imshow(wordcloud_tot)

# positive words # Choose the path for +ve words stored in system
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

# negative words  Choose path for -ve words stored in system
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
corpus_neg_in_neg = " ".join ([w for w in corpus_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
corpus_pos_in_pos = " ".join ([w for w in corpus_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)


# Unique words 
unique_words = list(set(" ".join(corpus_words).split(" ")))





#############################################

##############################################

df_Mounica = pd.read_csv('Mounica_chats.csv')

df_m= df_Mounica.iloc[:, [1]]
df_m.columns=["content"]
df_m.isnull().sum() #no na values
df_m.shape
df_m.info()
df_m.describe()
df_m.dtypes

# removing duplicate data
df_m.drop_duplicates(subset='content', keep='first', inplace=True)

# How long are the lenght of the contents

df_m['length'] = df_m['content'].map(lambda text: len(text))

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

df_m['content']= df_m["content"].apply(preprocessor)

#tokenizing and lemmatizing
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer() 

def tokenizer_lemmatizer(text):
    return[lem.lemmatize(word, "v") for word in text.split()]


from nltk.corpus import stopwords
stop= stopwords.words('english')

stop=stop + ['mounica','patel','ananya']


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
corpus_m = []

for i in df_m.index.values:
    chat_content_m=[w for w in tokenizer_lemmatizer(df_m['content'][i]) if w not in stop]
    
    # lem = WordNetLemmatizer()
    # df_m['content'] = [lem.lemmatize(word, "v") for word in df_m['content'] if not word in set(stop)]
    chat_content_m = ' '.join(chat_content_m)
    corpus_m.append(chat_content_m)


corpus_m= pd.DataFrame(corpus_m)
corpus_m.to_csv('mounica_corpus.csv', index='false')

corpus_m= pd.read_csv('mounica_corpus.csv')
corpus_m= corpus_m.dropna()
corpus_m.describe

corpus_m.drop_duplicates(keep='first', inplace=True)
# Joinining all the reviews into single paragraph 
corpus_m_string = " ".join(corpus_m.iloc[:, 0])
#all word tokens
words_tokens_m= word_tokenize(corpus_m_string)

corpus_m_words = corpus_m_string.split(" ")

# ### Counter

from collections import Counter

counter = Counter(words_tokens_m)
counter.most_common(20)

# # Bar Plot
# =============================================================================
# convert list of tuples into data frame
freq_df = pd.DataFrame.from_records(counter.most_common(20), columns =['Words_Token','Count'])

#Creating a bar plot
freq_df.plot(kind='bar',x='Words_Token', figsize=(15,10),fontsize=15);

# Simple word cloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud_tot = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_m_string)

plt.imshow(wordcloud_tot)

# positive words # Choose the path for +ve words stored in system
with open("positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

# negative words  Choose path for -ve words stored in system
with open("negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
corpus_m_neg_in_neg = " ".join ([w for w in corpus_m_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_m_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
corpus_m_pos_in_pos = " ".join ([w for w in corpus_m_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(corpus_m_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)


# Unique words 
unique_words = list(set(" ".join(corpus_m_words).split(" ")))


#################################################

#################################################

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=5,ngram_range =(1, 3)) 
X1 = vectorizer.fit_transform(words_tokens)  
features = list(vectorizer.get_feature_names())

counter = Counter(features)
counter = Counter(counter)
counter.most_common(20)
# convert list of tuples into data frame
freq_df_b = pd.DataFrame.from_records(counter.most_common(20), columns =['bigram','Count'])

#Creating a bar plot
freq_df_b.plot(kind='bar',x='bigram', figsize=(15,10),fontsize=15);


vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(words_tokens)

X1 = vect.transform(words_tokens)

features= list(X1.get_feature_names())
counter=Counter(vect.get_feature_names())

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
freq_df = pd.DataFrame.from_records(counter.most_common(50), columns =['Word_tokens','Count'])

#Creating a bar plot
freq_df.plot(kind='bar',x='Word_tokens', figsize=(15,10),fontsize=15);

freq_df = pd.DataFrame.from_records(counter_bigram.most_common(50), columns =['bigram_visitor','Count'])



# spelling correction

#all word tokens
#tokens= word_tokenize(" ".join(df_v.content))

from nltk.corpus import words

correct_spellings = words.words()

from nltk.metrics.distance import (
    jaccard_distance,
    )

from nltk.util import ngrams
spellings_series = pd.Series(correct_spellings)

correct = []
for entry in tokens :
    spellings = spellings_series[spellings_series.str.startswith(entry[0])]
    try:
        distances = ((jaccard_distance(set(ngrams(entry, 4)),set(ngrams(word, 4))), word) for word in spellings)
    except ZeroDivisionError:
        distances = 0
    #distances = ((jaccard_distance(set(ngrams(entry, 4)),set(ngrams(word, 4))), word) for word in spellings)
    closet = min(distances)
    correct.append(closet[1])



def answer_eleven(entries):
    from nltk.metrics.distance import (
    edit_distance,
    )
    spellings_series = pd.Series(correct_spellings)
    correct = []
    for entry in entries :
        spellings = spellings_series[spellings_series.str.startswith(entry[0])]
        distances = ((edit_distance(entry,word), word) for word in spellings)
        closet = min(distances)
        correct.append(closet[1])
        
    return correct
    
results= answer_eleven(words_tokens[0:100])
