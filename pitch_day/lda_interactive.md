![Demo](https://github.com/Erolino/Support_group_MD/blob/master/draft/logo.png)
# Introduction

## God forbid, 
##### you have a chronic health condition. 
Like everyone else the 1st place you go to, even before the doctor, is sometimes google, and from there on the route is very short to sinking deep into health forums and be overwhelmed in a sea of opinions. Is there a better way to make sense of all of this information?

"Support Group MD", is a project that takes a big step towards the goal of giving you health advice from the cumulative knowledge and experience of thousands of users who are fighting the same health battle. 

The pilot for "Support Group MD" is based on health forums that deal with a specific condition called: “Candida overgrowth”
##### “Candida overgrowth”, 
is a health condition that effects millions, who suffer from numerous symptoms and have no defined and consolidated treatments. This led to the emergence of numerous support forums of users advising each other on what treatment they tried, and what worked for them. The problem is that forums are highly disorganized, users get information overload, and are sometimes left to conduct meticulous research to get even the smallest hint for a potential solution.

So what if we could detect the best treatments that worked, from thousands of user experiences?

The ultimate goal of this project is to build a data base of treatments advised by users (by using NLP methodologies) and try to predict users wellbeing based on this data base (by using Sentiment Analysis and ML classification models).

![Demo](https://github.com/Erolino/Support_group_MD/blob/master/draft/capstoneTEST%20copy.jpg)
# Project Outline List
#### 1. Blog minning
    a. Webscarping - "The Candida forum", a case study
#### 2. Data processing
    a. General processing
    b. NLP text pre-processing 
#### 3. The Candida Forum Topics:
    a. Unsupervised clustering of corpus (with LDA topic distribution with pyLDAvis)
    b. Topics of Candida sentences - (with sent2topic()) 
#### 4. Semantic relations between Candida words (with word2vec):
    a. Dimentionality reduction and visualization (with bokeh)
    b. Associations with specific word examples (with get_related_terms(u'reaction',12))
    c. Word algebra with specific word combinations (with word_algebra(add=['gut','symptom'],subtract=['cure'],topn=15) (use specific cell))
#### 5. Towards minning of relevant words and then sentences (e.g. to "treatment"):
    a. Topic and Semantic Universe of a word (with ad hoc method: word_profile(word))
#### 6. Work in progress:
    a. Perfecting and using the "word_profile" method to extract relevant words,sentences (e.g. to "treatment")
    b. Sentiment analysis (why should we use it? + initial exploration)
    
## 2. The Candida Forum Topics:
#### a. Unsupervised clustering of corpus (with LDA topic distribution with pyLDAvis)
![Demo](https://user-images.githubusercontent.com/24357654/43748982-3f838ae6-99c0-11e8-8d1a-86b3cf7339ff.gif)
## Click: [LDA_interactive](https://nbviewer.jupyter.org/github/Erolino/Support_group_MD/blob/master/draft/pyLDAvis_10_topics.ipynb) to explore the different topics interactively!

#### b. Topics of Candida sentences - (with sent2topic()) 
>> sent2topic(93978)
out - "I think this could be the reason for me waking up middle of the night."
    - [[0, "daily do's and dont's", 0.47399999999999998],
      [1, 'ingredients/supplements', 0.20499999999999999]]

![Demo](https://user-images.githubusercontent.com/24357654/43749052-80f39f7a-99c0-11e8-8c37-11c176351aba.gif)

## Click: [t-SNE_interactive](https://nbviewer.jupyter.org/github/Erolino/Support_group_MD/blob/master/draft/t-sne_with_bokeh.ipynb) to explore word similarities between terms interactively


