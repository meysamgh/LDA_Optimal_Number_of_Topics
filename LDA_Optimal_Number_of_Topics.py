#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 17:57:30 2020

@author: Meysam
"""


import pandas as pd
from gensim.models import CoherenceModel

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
from gensim import corpora, models  # For TF-IDF
import matplotlib.pyplot as plt

from nltk import PorterStemmer  # Imported by myself

def lemmatize_stemming(text):
#    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v')) # Initial Code that had bug     
    return(PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v')))  # My updated version of code
    
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


data = pd.read_csv('reviews.csv', error_bad_lines=False);
data_text = data[['review']]
data_text['index'] = data_text.index
documents = data_text
    

processed_docs = documents['review'].map(preprocess)
processed_docs[:10]

dictionary = gensim.corpora.Dictionary(processed_docs)

# Filter rare words and words that appear in more than half of the documents. Keep first 100K most frequent ones
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)



bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

bow_doc_4310 = bow_corpus[100]
for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                               dictionary[bow_doc_4310[i][0]], 
bow_doc_4310[i][1]))
    
    
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint
for doc in corpus_tfidf:
    pprint(doc)
    break

#LDA:
c_vTF=[]
c_vBOW=[]


for i in range (2,20):
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(" NUMBER OF TOPICS:  ", i , " OUTPUT: ")
    
    tmpcoh=0
    for j in range (0,10):
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=i, id2word=dictionary, passes=6, workers=4)
        
        
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        tmpcoh += coherence_model_lda.get_coherence()
    c_vBOW.append(tmpcoh/10)
    

    
    for idx, topic in lda_model.print_topics(-1):
        print('BOW Topic: {} \nWords: {}'.format(idx, topic))    
        
        
    print("___________________________________________________")   
    # LDA on TF-IDF
    tmpcoh=0
    for j in range (0,10):    
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=i, id2word=dictionary, passes=6, workers=4)
        
     
        
        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        tmpcoh += coherence_model_lda.get_coherence()
    c_vTF.append(tmpcoh/10)    

    
    
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('TFIDF Topic: {} Word: {}'.format(idx, topic))

topics=[]
for i in range (2,20):
    topics.append(i)

# Plot with differently-colored markers.
plt.plot(topics, c_vBOW, 'b-', label='C_V Bag of Words')
plt.plot(topics, c_vTF, 'g-', label='C_V TF-IDF')

# Create legend.
plt.legend(loc='upper left')
plt.xlabel('Number of Topics')
plt.ylabel('C_V')
        
        
