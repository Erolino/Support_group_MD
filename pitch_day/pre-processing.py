#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 22:37:56 2018

@author: eran
"""
'''
1.2. Pre-processing and Saving the Data
     after retrieving the data, and saving into "scraped_raw.csv" ( look at candida_scraper.py),
     we need to concatenate different DFs and pre-process the posts, so we can follow users,posts 
     and sentences in an organized way 
'''

# paths to be used later for more ease:
from os.path import join as joinp
pitch_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pitch_day'  # thr directory with the final files of the project
process_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/processing_files'  # the directory with all the files used for NLP preprocessing and LDA modeling (.mm .txt etc)
support_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/' # the main directory of the project
pilot_path='/Users/eran/Galvanize_more_repositories/Support_group_MD/pilot' ## the directory to store files for this script

import pandas as pd
import numpy as np

## reading in the scraped data:
# Reading in from the 3 sources of scraped data  
dff=pd.read_csv(joinp(pitch_path,'discussion_raw.csv'))
print('number of rows(posts) =',len(dff))
dff2=pd.read_csv(joinp(pitch_path,'stories_raw.csv'))
dff3=pd.read_csv(joinp(pitch_path,'questions_raw.csv'))

## Let's merge the whole data together
## but before concatinating the 3 df, let's adjust the topic_ID
dff2['topic_ID']=dff2['topic_ID']+dff.iloc[-1,2]
dff3['topic_ID']=dff3['topic_ID']+dff2.iloc[-1,2]

#let's concatenate the dfs:
DF = pd.concat([dff, dff2, dff3])

# there are some nulls in the post_text data:
print(DF.info())
print('________________________')
print("any nulls in the data?",DF['post_text'].isnull().values.any())
print('number of nulls =',sum(DF['post_text'].isnull()))

# Very few nan, so let's drop them:
DF=DF.dropna()
print("any nulls in the data?",DF['post_text'].isnull().values.any())

# let's rearange a bit the DF
DF=DF.rename(index=str, columns={"Unnamed: 0": "ID_of_post_in_topic"})
DF=DF[['user_name','topic_ID','ID_of_post_in_topic','post_text']]
print("number of unique users (patients):",len(pd.unique(DF['user_name'])))

# let's make a more "descriptive" post id (than the index)
# we'll make a new column -"post_ID", by transforming the column - "ID_of_post_in_topic" into unique identifier
DF['post_ID']=DF['topic_ID']*100+DF['ID_of_post_in_topic']
DF=DF[['user_name','topic_ID','ID_of_post_in_topic','post_ID','post_text']]# arranging the coulmns again

print('are all post_ID unique?',len(DF['post_ID'].unique())==len(DF['post_ID']))
print('number of posts:',len(DF['post_ID']))

# Let's create a unique identifier instead of the user names for privacy purposes'
dic_user=dict(zip(DF['user_name'].unique(),range(len(DF['user_name'].unique()))))
## made a dictionary of {user_ID:user_name}
print(dic_user['Jeska']) # e.g.
DF['user_id']=DF['user_name'].transform(lambda x: dic_user[x])

''' Saving the data
    we want to have only user_id and not user name. so let's save a user to user_name table '''

# first let's save a user name to user Id table so we can refer to the names if we need.
y=pd.DataFrame(list(zip(DF['user_name'].unique(),range(len(DF['user_name'].unique())))),columns=['user_name','user_id'])
y.to_csv(joinp(pilot_path,'user_names_id.txt'))

# let's save data without user names:
DFF=DF[['user_id','topic_ID','ID_of_post_in_topic','post_ID','post_text']]
DFF.to_csv(joinp(pilot_path,'merged_W_id.txt'))
DFF.to_csv(joinp(pilot_path,'merged_W_id.csv'))

# The file we can refer to for user_names (in case we scrape more data)
print("user_names to user_id file:")
print(joinp(pilot_path,'user_names_id.txt'))

# the file we're going to work with for NLP-preprocessing!
print("the csv file, where the 'post_text' column is ready for NLP-preprocessing (A.K.A 'A'):")
joinp(pilot_path,'user_names_id.csv')

'''Next: NLP_pre-processing.py

