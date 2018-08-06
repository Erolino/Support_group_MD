#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 19:44:04 2018

@author: eran
"""

'From a normalized text file - extract topics using LDA (gensim)'

draft_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/draft'
pilot_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pilot'
pitch_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pitch_day'

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

if 1 == 0: ## no need to save this again
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
joinp(pilot_path, 'lda_model_10')

if 1 == 1:

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        # workers => sets the parallelism, and should be
        # set to your number of physical cores minus one
        lda = LdaMulticore(sent_bow_corpus,
                           num_topics=10,
                           id2word=sents_dict,
                           workers=1,dtype=np.float64)
if 1 == 0:    
    lda.save(joinp(pilot_path, 'lda_model_10')) ## no need to save again
    
# load the finished LDA model from disk
lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_10'))

''' Our topic model is now trained and ready to use! Since each topic is represented as a 
mixture of tokens, you can manually inspect which tokens have been grouped together into 
which topics to try to understand the patterns the model has discovered in the data.'''

def explore_topic(topic_number, topn=25, model=10):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
    #
    if model==25:
        lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_25'))
        topicname=topic_names_25[topic_number]
        gensimSTR=''
    elif model==15:
        lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_15'))
        topicname=topic_names_15[topic_number]
        gensimSTR=''
    elif model==10:
        lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_10'))
        topicname=topic_names_10[topic_number]
        gensimdic={0:9,1:8,2:6,3:7,4:3,5:10,6:5,7:1,8:2,9:4}
        gensimSTR=str(gensimdic[topic_number])
    
    
    #    
    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')
    
    dic={}
    j=0
    
    print('top 5 terms')
    for term, frequency in lda.show_topic(topic_number, topn):
        j=j+1
        if j<6:
            print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))
        dic[term]=frequency
    dff=pd.DataFrame.from_dict(dic,orient='index')
    dff.columns=[''.join(['topic:',topicname,' (gensim topic:',gensimSTR,')'])]    
    return(dff)
    ##
    
def explore_topic_nouns(topic_number, topn=25, model=10):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """
    #
    if model==10:
        lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_10'))
        topicname=topic_names_10[topic_number]
        gensimdic={0:9,1:8,2:6,3:7,4:3,5:10,6:5,7:1,8:2,9:4}
        gensimSTR=str(gensimdic[topic_number])
    
    #    
    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')
    
    dic={}
    j=0
    
    print('top 5 terms')
    for term, frequency in lda.show_topic(topic_number, topn):
        if dfff[dfff['nouns']==term].empty: ## dfff is loaded from pilot_path/bow_nouns.csv
            pass
        else:
            j=j+1
            if j<6:
                print (u'{:20} {:.3f}'.format(term, round(frequency, 3)))
            dic[term]=frequency
    dff=pd.DataFrame.from_dict(dic,orient='index')
    dff.columns=[''.join(['topic:',topicname,' (gensim topic:',gensimSTR,')'])]    
    return(dff)


'''##########'''
'''change topic names for numbers to labels that look descriptive. save topic names into topics: '''

topic_names_25 = {0:'0',1:'1',2: '2',3:'3',4: '4',5: '5',6: '6',7: '7',8: '8',9:'9', 10: '10',11:'11',12:'12',13:'13',14:'14',15:'15',16: '16',17: '17',18: '18',19: '19', 20: '20', 21: '21', 22: '22', 23: '23',24: '24'}

topic_names_15 = {0:'0',1:'1',2: '2',3:'3',4: '4',5: '5',6: '6',7: '7',8: '8',9:'9', 10: '10',11:'11',12:'12',13:'13',14:'14'}

topic_names_10 = {0:"daily do's and dont's",1:'ingredients/supplements',2: 'admin/courtesy',3:'dietary restriction /natural treatment',4:'medical problems manifestations',5: 'pathology / mechanism of action',6: 'food',7: 'experience / feeling / symptom',8: 'get rid / cure',9:'time'}
joinp(pilot_path, 'topic_names_10.pkl')

if 1==0: ## no need to save again
    with open(joinp(pilot_path, 'topic_names_10.pkl'), 'wb') as f:
        pickle.dump(topic_names_10, f)
    
'''##########'''

'''These are the topics I labeled:'''

with codecs.open(joinp(pilot_path, 'topic_names_10.pkl'), 'rb') as f:
    x = pickle.load(f)
x

explore_topic(topic_number=4, model=10)


''' VISUALIZATION  '''

#trigram_bow_filepath_draft = os.path.join(draft_path,'trigram_bow_corpus_all_draft.mm')
#sent_bow_corpus = MmCorpus(trigram_bow_filepath_draft)
#sents_dictionary = Dictionary(trigram_posts)

## old: LDAvis_data_filepath = os.path.join(draft_path,'ldavis_25_prepared')
joinp(pilot_path,'ldavis_10')

if 1 == 0:  ## for visualization, no need to run right now

    LDAvis_prepared = pyLDAvis.gensim.prepare(lda, sent_bow_corpus,
                                              sents_dictionary)

    with open(joinp(pilot_path,'ldavis_10'), 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
''' WHEN RUNNING THE NEXT LINES IN .ipynb, THERE WILL BE VISUALIZATION'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
''' (for visualization refer to "Candida_NLP_pyLDAvis_visualizing-sents") '''

# load the pre-prepared pyLDAvis data from disk
with codecs.open(joinp(pilot_path,'ldavis_10'),'rb') as f:  ## used 'rb' cause of this https://github.com/tkipf/gcn/issues/6
    LDAvis_prepared = pickle.load(f)
    
pyLDAvis.display(LDAvis_prepared)

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
''' let's continue with LDA analysis of specific sentences '''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

#Let's "import all the necessary variables and files needed to do this pipeline that are in the former: .ipynb and post_LDA.py files'

bigram_model = Phrases.load(joinp(pilot_path, 'bigram_model_all'))
trigram_model = Phrases.load(joinp(pilot_path, 'trigram_model_all'))
lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_10'))
# or lda_model_15 or lda_model_10
sents_dict = Dictionary(sents)
# and helper function:
def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """
    return token.is_punct or token.is_space


##old:
#trigram_dictionary_filepath_draft = os.path.join(draft_path,'trigram_dict_all_draft.dict')
#trigram_dictionary = Dictionary.load(trigram_dictionary_filepath_draft)
#lda_model_filepath_draft_25 = os.path.join(draft_path, 'lda_model_all_draft_25')
#lda = LdaMulticore.load(lda_model_filepath_draft_25)

# The Pipeline
# if you want to change the number of topics - load a different lda model:
# e.g. : lda = LdaMulticore.load(joinp(pilot_path, 'lda_model_25')) 
def lda_description(review_text, min_topic_freq=0.05,topic_model_file='lda_model_10'):
    """
    accept the original text of a review and (1) parse it with spaCy,
    (2) apply text pre-proccessing steps, (3) create a bag-of-words
    representation, (4) create an LDA representation, and
    (5) print a sorted list of the top topics in the LDA representation
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
    
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
    #print('bow:',trigram_review)
    
    # create a bag-of-words representation
    review_bow = sents_dict.doc2bow(trigram_review)
    
    
    # create an LDA representation
    lda = LdaMulticore.load(joinp(pilot_path, topic_model_file)) # my addition
    review_lda = lda[review_bow]
    
    
    # mine
    if topic_model_file=='lda_model_25':
        topic_names=topic_names_25
    elif topic_model_file=='lda_model_10':
        topic_names=topic_names_10
    #
    
    # sort with the most highly related topics first
    #review_lda = sorted(review_lda, key=lambda topic_number,freq: freq)
    listt=[]
    for topic_number, freq in review_lda:
        if freq < min_topic_freq:
            break
            
        # print the most highly related topic names and frequencies
        #print('{:10} {}'.format(topic_names[topic_number],round(freq, 3))) ## for now not putting yet topic names
        #print('{:25} {}'.format(topic_number,round(freq, 3))) 
        x=[topic_number,topic_names[topic_number],np.round(freq, 3)]
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
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
    with codecs.open(file_path, encoding='utf_8') as f:
        for sentence in it.islice(f,sent_number, sent_number+1):
            return sentence.replace('\n', '') 
        
## Let's get the raw file (this is me)
raw_path = joinp(pilot_path,'raw_sentences_all.txt')  
with codecs.open(raw_path,'r',encoding='utf8') as f:
    raw_sent = f.read()
    
## let's read a sample post:
sample_sent = get_sample_sent(raw_path,350)
print(sample_sent)

## this the topic distribution of the sample_sent:
y=lda_description(sample_sent,min_topic_freq=0.025,topic_model_file='lda_model_10')
y


## let's remeber what are these topics:
explore_topic(topic_number=8,model=10)
    
'''###################'''
''' added a helper functions to find specific words in the corpus by topic'''
## choosing 1000 words from a topic:

def findword(word,topicnum,rangee):
    p=lda.show_topic(topicnum,rangee)
    #print('p is',p)
    ## looking for oregano_oil
    x=None
    tup=[]
    rank=100000
    topicnumber=[]
    for ii in range(0,rangee):
        if p[ii][0]==word:
            x=' '.join([str(ii),'in topic',str(topicnum),topic_names_10[topicnum]])
            #print('rank=',str(ii))
            rank=ii
            topicnumber.append(topicnum)
            tup=p[ii][1]
    if x!=None:
        print(x)
    return(topicnumber,topic_names_10[topicnum],rank,tup)
    #return(tup,rank)
    
findword('drink',9,5)
explore_topic(topic_number=9,model=10)
    
''' to find a specific word in the corpus, print out the topic num, where is it ranked in the topic and what's its portion'''
## notice that the out put is/are the topics type - list
[iii for iii in range(0,9) if findword('candida',iii,5000)!=[]]
        

'''let's get the most common words:'''
tri=pd.read_csv(joinp(pilot_path, 'sentences_for_sentiment.txt'))
tri.head()
trilist=list(tri.iloc[:,0])
len(tri)

from collections import Counter     
Counter(trilist).most_common(5)
all=Counter(trilist)
## how many times candida is in the corpus
all['candida']

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~''' 
'''##### algorithm to find relevant sentences/words to treatment ###### '''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''   

''' strategy 1) sentence > topic of each word > how closely related to treatment:'''
        
sample_sent = get_sample_sent(raw_path,350) #3 sentence number 350
print(sample_sent)
## equivalent processed sentence (number 334):
prosent=tri.iloc[334,0] 
print(prosent)
## filter sentences by 3 chosen topics:
sent_topic=lda_description(sample_sent,min_topic_freq=0.05,topic_model_file='lda_model_10')
rel_sent=[]
for ii in range(0,len(sent_topic)):
    if sent_topic[ii][1]=='ingredients/supplements' or 'dietary restriction /natural treatment' or 'get rid / cure':
        t=True
if t==True:
    print(sample_sent)
# check which word is important: word
# another example:  prosent=tri.iloc[230,0]    
nlpprosent=nlp(prosent)
listprosent=[word for word in nlpprosent]
bow=[]
u=False
iii=[]
for word in listprosent:
    print(word)
    print(type(word))
    for iii in [1,3,8]:
        print(iii)
        if findword(str(word),iii,500)!=[]:
            u=True
            print('word is:',word,'in topic',iii)
    if u:
        bow.append(word)
    u=False

bow
 
''' strategy 2) 
    relevant topics of each word > how closely related to treatment:'''
''' 4 relevant topics: '''
# 1: 'ingredients/supplements',
# 3: 'dietary restriction /natural treatment',
# 4: 'medical problems manifestations',
# 8: 'get rid / cure'
''' get 500 most prevalent from each topic: '''
top4_500=explore_topic(topic_number=4, model=10,topn=500)
top4_500.tail() ## just to see
top1_500=explore_topic(topic_number=1, model=10,topn=500)
top3_500=explore_topic(topic_number=3, model=10,topn=500)
top8_500=explore_topic(topic_number=8, model=10,topn=500)

'''bow of only nouns:'''

if 1==0: ## this takes a long long time ~40 mins
    nouns=[]
    tags=[]
    for sent in trilist: # trigram_sentences: 
        nlpsent=nlp(sent)
        for term in nlpsent:
            if term.tag_ in ['FW','NN','NNS','NNP','NNPS']:
                nouns.append(term)
                tags.append(term.tag_)
        #else:
           # print('did not make it:',term,term.tag_)

if 1==0:  # this is to create bow of nouns, needed to do it (correctly!) only one time, and already did
    len(nouns) # 1,362,757  
    len(tags)         
    nouns[0:20] # long list of nouns   
    tags [0:20] # lomg list of tags (all should be nouns)
    myset=set(nouns) # get only unique
    bow_nouns=list(myset) ## finaly the bow of nouns
    
    '' 'to turn into bow (unique) and save the list to a file: '''
    
    df=pd.DataFrame(bow_nouns, columns=['nouns'])
    df.head()
    len(df)
    ## let's turn into unique:
    dfnew=df['nouns'].astype(str)
    len(dfnew)
    type(dfnew)
    valueC=dfnew.va
    dfuni=dfnew.unique() ## finally - 45278 unique nouns
    len(dfuni)
    dfff=pd.DataFrame(dfuni,columns=['nouns'])
    dfff.head()
    if 1==0: # no need to save this again
        dfff.to_csv(joinp(pilot_path,'bow_nouns.csv'))

''' word2Vec implementation '''
## read in df of words with highest similarities to antibiotics (created in word2vec Candida_NLP_word2vec.ipynb )
'''
## draft for next functions
antibW=pd.read_csv(joinp(pilot_path,'antib.csv'))
antibW=antibW.drop(['Unnamed: 0'],1)
antibW.head()
'''

def word2LDA(word):
    ranks=2000000
    ans=[]
    for hh in range(0,10):
        f=findword(word,hh,5000)
        #print('f',f)
        if f!=[]:
            #print('RANKS',ranks)
            t=f
            topicss=t[0]
            #print('topicss',topicss)
            topicnames=t[1]
            prob_in_topic=t[3]
            #print('t[2]',t[2])
            #print(type(int(t[2])))
            #print(type(ranks))
            if int(t[2])<ranks:
                ranks=t[2]
                ans=t
                #print('ans',ans)
    return(ans)

# word2LDA('antibiotics') ## e.g. gives a topic profile of a specific word   

## create a df of an LDA profile of words with highest similarities to antibiotics 



curedf=pd.read_csv(joinp(pilot_path,'cure.csv'))

def LDA_to_df(col_of_words): 
    ##e.g. col_of_words=antibW['words'] 
    df2=pd.DataFrame(columns=['topic_num','topic_label','topic_rank','topic_score'])
    df3=pd.DataFrame(columns=['topic_num','topic_label','topic_rank','topic_score'])
    df2.head()
    for ii,word in enumerate(col_of_words):
        row=word2LDA(word)
        df2=pd.DataFrame([row],columns=['topic_num','topic_label','topic_rank','topic_score'])
        df3=pd.concat([df3,df2])
    
    dfLDA=df3
    return (dfLDA)

cureLDA=LDA_to_df(curedf['words'])
    
''' concatenate word2vec data and LDA data into 1 df for analysis and visualization'''
def df2df(dfW,dfLDA):
    #dfW=dfW.drop(['Unnamed: 0'],1)
    dfW.reset_index(drop=True, inplace=True) ## reseting indices otherwise can't concatenate
    dfLDA.reset_index(drop=True, inplace=True)
    final=pd.concat([dfW,dfLDA],axis=1)
    return (final)

finaldf=df2df(curedf,cureLDA)
finaldf.head()
if 1==0: ## if you want to save
    finaldf.to_csv(joinp(pilot_path,'cureWLDA.csv'))
'''
## this is to df2df draft 
antibW.reset_index(drop=True, inplace=True) ## reseting indices otherwise can't concatenate
antibLDA.reset_index(drop=True, inplace=True)
ANTIB=pd.concat([antibW,antibLDA],axis=1)
ANTIB.tail()
#let's save this one:
if 1==0: ## no need to save again
    ANTIB.to_csv(joinp(pilot_path,'antibW2LDA.csv'))
'''
# we can continue to analyze and visualize on Candida_NLP_word2vec.ipynb    

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~''' 
'''############### end of algorithm ##################### '''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''  



''' ##########  sentiment analysis   #################'''

'''let's put it aside for now
# let's try this trigram transformed sentence:
tri.iloc[230,0]
# look for equivalent pre-processed sentence
for ii in range(0,20):
    print(233+ii,get_sample_sent(raw_path,233+ii))
# found it:
get_sample_sent(raw_path,241)

## from https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis:
from textblob import TextBlob
## testing 1,2,3..:
testimonial = TextBlob("Textblob is amazingly simple to use. What great fun!")
testimonial.sentiment
testimonial.sentiment.polarity
## works!
bad=TextBlob(get_sample_sent(raw_path,244)) ##notes are for get_sample_sent(raw_path,241) 
bad.sentiment
bad.sentiment.polarity
## works for mine too!
## now try to test how good it is with other examples
joinp(pilot_path,'raw_sentences_all.txt') ## the path
rawpd=pd.read_csv(joinp(pilot_path,'raw_sentences_all.txt'),sep='<->',names=['sentence'])
rawpd.head() 
for sent in rawpd['sentence'][300:315]:
    sentblob=TextBlob(sent)
    sentblob.sentiment
    print(sent,sentblob.sentiment.polarity)

## conclusion:
## not very good. 1. accuracy is low e.g.: "Honey might be another option, 
## as it’s anti-bacterial, but since we’re dealing with a yeast that’s 
## gone wildly out of control, I would think it might be best to avoid even 
## honey at this point. 0.55 2. seems biased towards words 
## without it's content e.g. "Are anti-fungals effective at 
## eliminating candida in the prostate? 0.6" 
## 3. sentiment analysis might not be the answer as advice although valuable:
## e.g. doesn't have a lot of sentiment
## also FP - "irrelivent sentiment" is picked up. e.g.:     
    
''' #######  end of sentiment analysis   ###############'''


''' ########### draft ##################3'''

''' filter out unwanted entities using spacy and textblob '''
'''
ind4=top4_500.index
listW=list(ind)
listW[0:5]
strW=' '.join(listW)
blobW=TextBlob(strW)
blobTag=blobW.tags
len(blobTag) # blobTag[0]
# filter by tag =nouns 
#FW foreign word (e.g. mea culpa)
#NN noun, (e.g. sing. or mass llama) 
#NNS noun, plural (e.g. llamas) 
#NNP proper noun, (e.g. sing. IBM) 
#NNPS proper noun, (plural e.g. Carolinas)
filtered4=[]
for uu in range(0,20):
    print('blobTag[uu][1]=',blobTag[uu][1])
    if blobTag[uu][1] in ['FW','NN','NNS','NNP','NNPS']:
        print('blobTag[uu][0]=',blobTag[uu][0])
        filtered4.append(blobTag[uu][0])
        print(filtered4)
    else:
        print('#')
        print('blobTag[uu][0]=',blobTag[uu][0])
        print('#')
len(filtered4)
filtered4# [0:20]
blobTag[50:70]
blobTag[1][1]=='NNPS'


# testing difference in taging between raw and processed:
raw_sent[0:72] ## ''HiIs coconut nectar and coconut butter ok to have when treating candida?
pro_sent=get_sample_sent(joinp(pilot_path,'sentences_for_sentiment.txt'),0)
pro_sent ## hii coconut_nectar coconut_butter ok treat candida
pro_sent=nlp(pro_sent)
raw_sent_72=nlp(raw_sent[0:72])
pro=[]
raw=[]
for token in raw_sent_72:
    print(token)
    raw.append(token.tag_)
for token in pro_sent:
    print(token)
    pro.append(token.tag_)
raw
pro

trilist[0]    
'''
