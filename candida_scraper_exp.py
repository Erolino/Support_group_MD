#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 18:40:04 2018

@author: eran
"""


import os
os.chdir('/Users/eran/Jupyter notebooks/SIBO_project')
## this is a scraper script for the forum section in the site - The CandidaDiet.com
## the forum section has ~9120 topics (posts and comments)
url="https://www.thecandidadiet.com/forums/forum/general-discussion/"

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
import pandas as pd
import numpy as np
import time
from urllib.error import HTTPError

# 1. getting a webpage (type = str):
def getpage(youRL):
    from urllib.request import Request, urlopen
    try:
        req = Request(youRL, headers={'User-Agent': 'Chrome/63.0.3239.108'})
        web_byte = urlopen(req).read()
        webpage = web_byte.decode('utf-8')
        if webpage is None: 
            print('page is None')
        return webpage
    except (IndexError,HTTPError):
        while topic is None:
                jj=jj+1
                print('trying to fetch by refreshing',jj,'times')
                time.sleep(10)
                topic=getpage(link)
                if jj>10:
                    print('could not get the page for',link) 
                    print('moving on to next links')      
                    break
        
        'failed to get page with "fancy request"'
    except HTTPError:
       # print('there is an HTTPError inside "getpage", returnning None')
       # return None

# 2. getting the links for topics on a specific page
def pagelinks(page):
    bsob=soup(page,"html.parser") # creates a bs object from the mane page: "home›The Candida Forum›Candida Questions"
    topic_links_W_tags=bsob.select('a.bbp-topic-permalink')
    topic_links_W_tags[0]
    links=[link.get('href') for link in topic_links_W_tags] 
    if links[0]=='https://www.thecandidadiet.com/forums/topic/useful-links-and-forum-posts-25/':
        del links[0] ## deleting the 1st link (only on the 1st page) cause it's irrelevant 
    return links

# 3. once entering the links into a topic (a post and its comments), this function get the 
#    user names and what they wrote and outputs a dataframe
def getUserPost(page):
    #post section:
    bsob=soup(page,"html.parser")
    post_text_w_tag=bsob.find_all("div",class_="bbp-reply-content")#gdbbx-quote-wrapper-176128")
    post_text=[post.get_text() for post in post_text_w_tag]
    del post_text[0],post_text[-1] # deleting the 1st and last entry cause they return "posts"
    
    #user section:
    user=[]
    user_name_w_tag=bsob.find_all("div",class_="bbp-reply-author")
    for item in user_name_w_tag:
        user_parent=item.find_all("a",class_="bbp-author-name")
        if user_parent==[]:
            pass
        else:
            username=user_parent[0].get_text()
            user.append(username)
    
    df=pd.DataFrame({'user_name':user,'post_text':post_text}) 
    df=df[['user_name','post_text']]
    return df 

# 4. given page, outputs the link for the next page
def nextPageLink(page):
    bsob=soup(page,"html.parser")
    next_page_links_W_tag=bsob.find('a',class_='next page-numbers')
    link=next_page_links_W_tag.get('href')
    return link
    

# 5. the "mother" function - the actual scraper that combines all the above functions
def motherF(youRhell=None):    
    # let's measure the time it takes to run this scraper
    start_time = time.time()
    # make empty df to put all the data in it:
    col_names =  ['user_name','topic_ID', 'post_text']
    df0=pd.DataFrame(columns = col_names)
    #    " get the 1st page with links {using getpage()}"
    ii=0
    topicNum=0
    while ii<11:
        try:
            page=getpage(youRhell)
        except HTTPError as e:
            break
        ii=ii+1
        print('downloading',ii,"pages out of 610")
        links=pagelinks(page)
        for link in links:
            time.sleep(1.2)
            topic=getpage(link)
            jj=0
            while topic is None:
                jj=jj+1
                print('trying to fetch by refreshing',jj,'times')
                time.sleep(10)
                topic=getpage(link)
                if jj>10:
                    print('could not get the page for',link) 
                    print('moving on to next links')      
                    break
            try:
                df=getUserPost(topic)
                topicNum=topicNum+1
                df['topic_ID']=topicNum
                #concatenate the df to the big df:
                frames = [df0, df]
                df0 = pd.concat(frames)
            except TypeError as e:
                print('can not getUserPost from None')
                break
        try:
            time.sleep(1.2) # to not scrape it too fast
            youRhell=nextPageLink(page)
        except (AttributeError,HTTPError):
            break
    print("scraping time =",round(time.time() - start_time,2),"sec")
    # 2 secs X 610 pages is 610 sec so substract that..
    return df0[['user_name','topic_ID', 'post_text']]
        
            
    
                     
# 'To operate the scraper and save the data frasme to a csv:'

scrapeddf=mother('https://www.thecandidadiet.com/forums/forum/general-discussion/')
#scrapeddf.to_csv('/Users/eran/Jupyter notebooks/SIBO_project/scraped1.csv')

len(scrapeddf)


shit='https://www.thecandidadiet.com/forums/forum/general-discussion/pagee/100000/'
good='https://www.thecandidadiet.com/forums/forum/general-discussion/page/100/'

u=getpage(shit)
t=0
while u is None:
    time.sleep(1)
    u=getpage(shit)
    print("u is shit")
    if t==3:
        u=getpage(good)
        print('u is good')
    elif t==6:
        break
    t=t+1
print('it continued')

getUserPost(None)
