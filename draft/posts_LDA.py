#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:44:04 2018

@author: eran
"""

'From a normalized text file - extract topics using LDA (gensim)'

draft_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/draft'
pilot_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pilot'

import pandas as pd
import numpy as np
from os.path import join as joinp
import codecs
import itertools as it
import spacy
import boto3
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

nlp = spacy.load('en')

spacy.lang.en.STOP_WORDS.add("-PRON-")
spacy.lang.en.STOP_WORDS.add("'")

## this is the post-"preprocessing" text file done in previous steps (after normalizing,lematizing, trigraming and rm stopwords)
print(joinp(pilot_path,'sentences_for_sentiment.txt'))
# t=pd.read_csv(joinp(pilot_path,'sentences_for_sentiment.txt'))

## let's load 4 sentences:
with codecs.open(joinp(pilot_path,'sentences_for_sentiment.txt'), encoding='utf_8') as f:
    for sent in it.islice(f, 0, 4):
        print(sent)

from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis
import pyLDAvis.gensim
import warnings
import pickle


" LDA PART :" 

## the file path to save the dictionary into:
joinp(pilot_path,'sent_dict.dict')
# this is a bit time consuming - make the if statement True
# if you want to learn the dictionary yourself.
if 1 == 1:

    #old: trigram_posts = LineSentence(joinp(pilot_path,'sent_dict.dict'))
    sents = LineSentence(joinp(pilot_path,'sentences_for_sentiment.txt'))

    # learn the dictionary by iterating over all of the reviews
    #trigram_dictionary = Dictionary(sents)
    sents_dict = Dictionary(sents)
    
    
    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    sents_dict.filter_extremes(no_below=10, no_above=0.4)
    sents_dict.compactify()

    sents_dict.save(joinp(pilot_path,'sent_dict.dict'))
    
# load the finished dictionary from disk
sents_dictionary = Dictionary.load(joinp(pilot_path,'sent_dict.dict'))

'''The sents_bow_generator function implements this. We'll save the 
resulting bag-of-words reviews as a matrix.'''


# trigram_bow_filepath_draft = os.path.join(draft_path,'trigram_bow_corpus_all_draft.mm')
joinp(pilot_path,'sent_bow_corpus.mm')

def sents_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    
    for sents in LineSentence(filepath):
        yield sents_dictionary.doc2bow(sents)
        
if 1 == 1:

    # generate bag-of-words representations for
    # all posts and save them as a matrix
    MmCorpus.serialize(joinp(pilot_path,'sent_bow_corpus.mm'),sents_bow_generator(joinp(pilot_path,'sentences_for_sentiment.txt')))
    
# load the finished bag-of-words corpus from disk
# old: trigram_bow_corpus = MmCorpus(joinp(pilot_path,'sent_bow_corpus.mm'))
sent_bow_corpus = MmCorpus(joinp(pilot_path,'sent_bow_corpus.mm'))

''' With the bag-of-words corpus, we're finally ready to learn our topic model from the reviews. 
We simply need to pass the bag-of-words matrix and Dictionary from our previous steps to 
LdaMulticore as inputs, along with the number of topics the model should learn. For 
this demo, we're asking for 50 topics.'''

## old: lda_model_filepath_draft_25 = os.path.join(draft_path, 'lda_model_all_draft_25')
joinp(pilot_path, 'lda_model_25')

if 1 == 1:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # workers => sets the parallelism, and should be
        # set to your number of physical cores minus one
        lda = LdaMulticore(sent_bow_corpus,
                           num_topics=25,
                           id2word=sents_dict,
                           workers=1,dtype=np.float64)
    
    lda.save(joinp(pilot_path, 'lda_model_25'))
    
# load the finished LDA model from disk
lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_25'))

''' Our topic model is now trained and ready to use! Since each topic is represented as a 
mixture of tokens, you can manually inspect which tokens have been grouped together into 
which topics to try to understand the patterns the model has discovered in the data.'''

def explore_topic(topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
        
    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for term, frequency in lda.show_topic(topic_number, topn=25):
        print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))

explore_topic(topic_number=10)


'''##########'''
'''to change topic names.. save topic names into topics '''

topic_names = {0:'?',1:'?',2: 'medical',3:'supplements',4: 'looking for',5: 'carbs',6: 'yogurt',7: 'coffee tea food good',8: 'problem, need advice',9:'?', 10: 'protein diet',11:'?',12:'?',13:'admin',14:'?',15:'gut',16: 'feeling',17: 'thomas enema',18: 'buy',19: 'time regiment', 20: '?', 21: 'test/doc', 22: '?', 23: 'treatment',24: 'food, recipe'}

topic_names_filepath_draft_25 = os.path.join(draft_path, 'topic_names_draft_25.pkl')

with open(topic_names_filepath_draft_25, 'wb') as f:
    pickle.dump(topic_names, f)
    
'''##########'''

'''These are the topics I labeled:'''

with codecs.open(topic_names_filepath_draft_25, 'rb') as f:
    x = pickle.load(f)
x
''' VISUALIZATION  '''

#trigram_bow_filepath_draft = os.path.join(draft_path,'trigram_bow_corpus_all_draft.mm')
#sent_bow_corpus = MmCorpus(trigram_bow_filepath_draft)
#sents_dictionary = Dictionary(trigram_posts)

## old: LDAvis_data_filepath = os.path.join(draft_path,'ldavis_25_prepared')
joinp(pilot_path,'ldavis_25')

if 1 == 1:

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda, sent_bow_corpus,
                                              sents_dictionary)

    with open(joinp(pilot_path,'ldavis_25'), 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
''' WHEN RUNNING THE NEXT LINES IN .ipynb, THERE WILL BE VISUALIZATION'''
# load the pre-prepared pyLDAvis data from disk
with codecs.open(joinp(pilot_path,'ldavis_25'),'rb') as f:  ## used 'rb' cause of this https://github.com/tkipf/gcn/issues/6
    LDAvis_prepared = pickle.load(f)
    
pyLDAvis.display(LDAvis_prepared)
