#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 21:00:02 2018

@author: eran
"""

## import packages:
import pandas as pd
import numpy as np
from os.path import join as joinp
import itertools as it
import codecs


import spacy
nlp = spacy.load('en')

import boto3
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence

# Helper functions (from yelp..):

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    return token.is_punct or token.is_space

def line_review(list_of_posts):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    for post in list_of_posts:
        yield post.replace('\\n', '\n')

def lemmatized_sentence_corpus(list_of_posts):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """  
    for parsedPosts in nlp.pipe(line_review(list_of_posts)):
    
        for sent in parsedPosts.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])
    
       

pitch_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pitch_day'  # thr directory with the final files of the project
process_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/processing_files'  # the directory with all the files used for NLP preprocessing and LDA modeling (.mm .txt etc)
support_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/' # the main directory of the project
pilot_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pilot' ## the directory to store files for this script

## getting the original posts ('A')        
DFF=pd.read_csv(joinp(pitch_path,'merged_W_id.csv'))    
A=list(DFF['post_text']) ## just choosing the 1st 3 posts to explore

'''
    2. TEXT-PREPROCESSING
##############################'''

''' 2.1. Segment text of posts into sentences & normalize text (into Unigram) '''
'''~~~~~~~~~~~~~~'''
## now I can feed "A" into the function that parses,lenmatizes and gives out sentences
lsc=lemmatized_sentence_corpus(A) 

# writing a Unigram sentences file
# make the if statement True if you want to execute data prep yourself.( be careful for not overrunning files)
if 1 == 1:
    with codecs.open(joinp(pilot_path,'unigram_sentences_all.txt'), 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus(A):
            f.write(sentence + '\n')

unigram_sentences = LineSentence(joinp(pilot_path,'unigram_sentences_all.txt'))

''' 2.2. Bigrams - first-order phrase modeling →→ apply first-order phrase model to transform sentences '''
'''~~~~~~~~~~~~~~'''
### making a "Model" for bigram phrases
if 1 == 0:

    bigram_model = Phrases(unigram_sentences)
    bigram_model.save(joinp(pilot_path, 'bigram_model_all'))
    
# load the finished model from disk
bigram_model = Phrases.load(joinp(pilot_path, 'bigram_model_all'))

### fitting the unigram text to the bigram "Model" (and writing it to a file)
if 1 == 0:

    with codecs.open(joinp(pilot_path,'bigram_sentences_all.txt'), 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences: 
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + '\n')

bigram_sentences = LineSentence(joinp(pilot_path,'bigram_sentences_all.txt'))

''' 2.3. Trigrams - 2nd-order phrase modeling →→ apply 2nd-order phrase model to transform sentences '''
'''~~~~~~~~~~~~~~'''
### making a "Model" for Trigram phrases
if 1 == 0:

    trigram_model = Phrases(bigram_sentences)
    trigram_model.save(joinp(pilot_path,'trigram_model_all'))
    
# load the finished model from disk
trigram_model = Phrases.load(joinp(pilot_path,'trigram_model_all'))

### fitting the unigram text to the bigram "Model" (and writing it to a file)
if 1 == 0:

    with codecs.open(joinp(pilot_path,'trigram_sentences_all.txt'), 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences: 
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')

trigram_sentences = LineSentence(joinp(pilot_path,'trigram_sentences_all.txt'))
    
'''2.4. Apply text normalization (mainly removing STOP WORDS '''
'''~~~~~~~~~~~~~~'''
## adding ' and -PRON- to stop words
spacy.lang.en.STOP_WORDS.add("-PRON-")
spacy.lang.en.STOP_WORDS.add("'")
spacy.lang.en.STOP_WORDS.add("’")
spacy.lang.en.STOP_WORDS.add("-")

# this takes the trigram and does final normailzation (removing stop words). 
# the final file is sentence in every row
if 1 == 1:

    with codecs.open(joinp(pilot_path,'trigram_trans_sentences_all.txt'), 'w', encoding='utf_8') as f:
        for trigram_sent in trigram_sentences: 
            trigram_transformed = [term for term in trigram_sent
                              if term not in spacy.lang.en.STOP_WORDS]
            trigram_transformed = u' '.join(trigram_transformed)
            f.write(trigram_transformed + '\n')

trigram_transformed=LineSentence(joinp(pilot_path,'trigram_trans_sentences_all.txt'))
 
# we want to have no gaps between sentences, so we can just read in with pandas (automatically ignores empty rows:
## reading in the the text file:
trans_with_gaps=pd.read_csv(joinp(pilot_path,'trigram_trans_sentences_all.txt'),sep='<->',names=['sentence'])

''' Finally writing the text into a file, ready for LDA and sentiment anaslysis:'''
trans_with_gaps.to_csv(joinp(pilot_path,'sentences_for_sentiment.txt'), header=None, index=None, mode='w')









''' 
with codecs.open(joinp(yet_path,'trigram_transformed_sents_all.txt'), 'w', encoding='utf_8') as f:
        
    for parsed_p in nlp.pipe(line_review(A)):
            
            # lemmatize the text, removing punctuation and whitespace
        unigram_p = [token.lemma_ for token in parsed_p
                              if not punct_space(token)]
            
            # apply the first-order and second-order phrase models
        bigram_p = bigram_model[unigram_p]
        trigram_p = trigram_model[bigram_p]
            
            
            # remove any remaining stopwords
        trigram_p = [term for term in trigram_p
                              if term not in spacy.lang.en.STOP_WORDS]
            
            # write the transformed review as a line in the new file
        trigram_p = u' '.join(trigram_p)
        f.write(trigram_p + '\n')

'''