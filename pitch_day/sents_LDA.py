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
'''change topic names for numbers to labels that look descriptive. save topic names into topics: '''

topic_names = {0:'0',1:'1',2: '2',3:'3',4: '4',5: '5',6: '6',7: '7',8: '8',9:'9', 10: '10',11:'11',12:'12',13:'13',14:'14',15:'15',16: '16',17: '17',18: '18',19: '19', 20: '20', 21: '21', 22: '22', 23: '23',24: '24'}

joinp(pilot_path, 'topic_names_25.pkl')

with open(joinp(pilot_path, 'topic_names_25.pkl'), 'wb') as f:
    pickle.dump(topic_names, f)
    
'''##########'''

'''These are the topics I labeled:'''

with codecs.open(joinp(pilot_path, 'topic_names_25.pkl'), 'rb') as f:
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

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
''' let's continue with LDA analysis of specific sentences '''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

#Let's "import all the necessary variables and files needed to do this pipeline that are in the former: .ipynb and post_LDA.py files'

bigram_model = Phrases.load(joinp(pilot_path, 'bigram_model_all'))
trigram_model = Phrases.load(joinp(pilot_path, 'trigram_model_all'))
lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_25'))
sents_dict = Dictionary(sents)


##old:
#trigram_dictionary_filepath_draft = os.path.join(draft_path,'trigram_dict_all_draft.dict')
#trigram_dictionary = Dictionary.load(trigram_dictionary_filepath_draft)
#lda_model_filepath_draft_25 = os.path.join(draft_path, 'lda_model_all_draft_25')
#lda = LdaMulticore.load(lda_model_filepath_draft_25)

# The Pipeline
# if you want to change the number of topics - load a different lda model:
# e.g. : lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_25')) 
def lda_description(review_text, min_topic_freq=0.05,topic_model_file='lda_model_25'):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """
    
    # parse the review text with spaCy
    parsed_review = nlp(review_text)
    
    # lemmatize the text and remove punctuation and whitespace
    unigram_review = [token.lemma_ for token in parsed_review
                      if not punct_space(token)]
    
    # apply the first-order and secord-order phrase models
    bigram_review = bigram_model[unigram_review]
    trigram_review = trigram_model[bigram_review]
    
    # remove any remaining stopwords
    trigram_review = [term for term in trigram_review
                      if not term in spacy.lang.en.STOP_WORDS]
    
    # create a bag-of-words representation
    review_bow = sents_dict.doc2bow(trigram_review)
    
    # create an LDA representation
    lda = LdaMulticore.load(joinp(pilot_path, topic_model_file)) # my addition
    review_lda = lda[review_bow]
    
    # sort with the most highly related topics first
    #review_lda = sorted(review_lda, key=lambda topic_number,freq: freq)
    listt=[]
    for topic_number, freq in review_lda:
        if freq < min_topic_freq:
            break
            
        # print the most highly related topic names and frequencies
        print('{:10} {}'.format(topic_names[topic_number],round(freq, 3))) ## for now not putting yet topic names
        #print('{:25} {}'.format(topic_number,round(freq, 3))) 
        x=[topic_names[topic_number],np.round(freq, 3)]
        listt.append(x)
    return(listt)

'''############# Interlude ################'''

''' let's parse the raw posts into raw sentences (haven't done that yet, only have parsed unigram and above)'''

DFF=pd.read_csv(joinp(pitch_path,'merged_W_id.csv'))    
A=list(DFF['post_text']) ## just choosing the 1st 3 posts to explore

### let's make a parsing function (like lemmatized_sentence_corpus just without lemitizing) that uses 'line_review':

def sentence_parse(list_of_posts):
    """
    generator function to use spaCy to parse reviews and yield sentences
    """  
    for parsedPosts in nlp.pipe(line_review(list_of_posts)):
        for sent in parsedPosts.sents:
            yield str(sent)

### let's write it into a file:
if 1 == 0:
    with codecs.open(joinp(pilot_path,'raw_sentences_all.txt'), 'w', encoding='utf_8') as f:
        for sentence in sentence_parse(A):
            f.write(sentence + '\n')

'''#########################################'''

'''now we can continue with LDA getting topics inside every sentence:'''
    
def get_sample_sent(file_path,sent_number):
    """
    retrieve a particular review index
    from the reviews file and return it
    """
    with codecs.open(file_path, encoding='utf_8') as f:
        for sentence in it.islice(f,sent_number, sent_number+1):
            return sentence.replace('\n', '') 
        
## Let's get the raw file (this is me)
raw_path = joinp(pilot_path,'raw_sentences_all.txt')  
with codecs.open(raw_path,'r',encoding='utf8') as f:
    raw_sent = f.read()
    
## let's read a sample post:
sample_sent = get_sample_sent(raw_path,17)
print(sample_sent)

## this the topic distribution of the sample_sent:
y=lda_description(sample_sent)
topic_num1=int(y[0][0]) # 1st topic
topic_num2=int(y[1][0]) # 2nd topic
topic_num3=int(y[2][0]) # 3rd
## let's remeber what are these topics:
explore_topic(topic_number=topic_num3)
    
