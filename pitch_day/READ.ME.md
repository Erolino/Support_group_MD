![Demo](https://github.com/Erolino/Support_group_MD/blob/master/draft/logo.png)

## <p align="center">By Eran Schenker</p>

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

# Project Outline

![Demo](https://github.com/Erolino/Support_group_MD/blob/master/draft/capstoneTEST%20copy.jpg)
# Project Outline List
#### 1. Forum minning
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

## 1. Blog minning
#### a. Webscarping - "The Candida forum", a case study
![Demo](https://user-images.githubusercontent.com/24357654/43795581-cd99a934-9a4f-11e8-8546-ca7c656bc199.png)
Have a look at [The Candida Website](https://www.thecandidadiet.com/forums/)

Refer to [candida_scraper.py](https://github.com/Erolino/Support_group_MD/blob/master/pitch_day/candida_scraper.py) for the webscraper code
    
## 2. Data processing
#### a. General processing
![Demo](https://user-images.githubusercontent.com/24357654/43796549-8f507d94-9a52-11e8-83d2-fe9006f7d28f.png)
Refer to [pre-processing.py](https://github.com/Erolino/Support_group_MD/blob/master/pitch_day/pre-processing.py) for the code

    b. NLP text pre-processing

Refer to [NLP_pre-processing.py](https://github.com/Erolino/Support_group_MD/blob/master/pitch_day/NLP_pre-processing.py) for the code

## 3. The Candida Forum Topics:
#### a. Unsupervised clustering of corpus (with LDA topic distribution with pyLDAvis)

![Demo](https://user-images.githubusercontent.com/24357654/43748982-3f838ae6-99c0-11e8-8d1a-86b3cf7339ff.gif)
## Click: [LDA_interactive](https://nbviewer.jupyter.org/github/Erolino/Support_group_MD/blob/master/draft/pyLDAvis_10_topics.ipynb) to explore the different topics interactively!

Refer to sent_LDA.py for the code

#### b. Topics of Candida sentences - (with sent2topic()) 
>> sent2topic(93978)

>> out - "I think this could be the reason for me waking up middle of the night."
    - [[0, "daily do's and dont's", 0.47399999999999998],
      [1, 'ingredients/supplements', 0.20499999999999999]]

## 4. Semantic relations between Candida words (with word2vec):
#### a. Dimentionality reduction and visualization (with bokeh)
![Demo](https://user-images.githubusercontent.com/24357654/43749052-80f39f7a-99c0-11e8-8c37-11c176351aba.gif)

## Click: [t-SNE_interactive](https://nbviewer.jupyter.org/github/Erolino/Support_group_MD/blob/master/draft/t-sne_with_bokeh.ipynb) to explore word similarities between terms interactively

#### b. Associations with specific word examples (with get_related_terms())

By typing the word "doctor", we can see the types of doctors users are talking about:

>> get_related_terms(u'doctor',12)

>> out - similarity with: "doctor"

>> "specialist"	0.746734

>> "naturopath"	0.726649

>> "allergist"	0.718856

#### c. Word algebra with specific word combinations (with word_algebra(add=['__','__'],subtract=['__'],topn='__')

with word algebra we can explore gut symptoms and "exclude" terms that are related to cure

>> gut + symptom - cure:

>> "gas_bloating"

>> "gas"

>> "digestive_tract"

or vice versa - explore "gut cures" while filtering out terms that are related to symptoms:

>> gut + cure - symptom:

>> "repopulate"

>> "intestine"

>> "beneficial_flora"

So essentialy, the 3 most related terms to cure this condition is to repopulate the intestine with beneficial flora (i.e. "good bacteria")

## 5. Towards minning of relevant words and then sentences (e.g. to "treatment"):
#### a. Topic and Semantic Universe of a word (with ad hoc method: word_profile(word))
    
Let's type any word, and get:

### * Table
### * Bar plot
### * scatter plot

____________________________
### let's type "antifungal" and run the function:
### Table:

'words' - 30 most related terms 

'similarity with: _____' - euclidian distance, of the terms after running word2vec model

'topic_label' - the topics where these 30 terms have the highest rank after running LDA model

'topic_rank' - the actual rank in the topic

'topic_score' - the probability to find the term in that topic (prevalence)

![Demo](https://user-images.githubusercontent.com/24357654/43794790-6561ca24-9a4d-11e8-81e6-cad943a413c1.png)

### Bar plot:

A histogram of the terms count per topic ( how many in each topic)

this will give a sense of how accurate the LDA model is (e.g. we will expect terms related to cure to be in the 'get rid / cure' topic and less in the 'admin / coutesy' topic)

![Demo](https://user-images.githubusercontent.com/24357654/43794813-7c01e61a-9a4d-11e8-9530-583ab68a5a92.png)

### scatter plot:

How prevalent vs how related a term is to the word. the more to the right, the stronger the relationship, the higher up the more prevalent in its topic.

![Demo](https://user-images.githubusercontent.com/24357654/43794840-93a01094-9a4d-11e8-9fff-93035bb6a0b1.png)

## 6. Work in progress:
#### a. Perfecting and using the "word_profile" method to extract relevant words,sentences (e.g. to "treatment")

word profile could be used as folllows:

1. Before using "word_profile()", The BOW of the whole corpus could be filtered with an outside vocabulary

2. A vocabulary of drugs, supplements and ingridients should be used to keep useful entities and filter out general nouns that are not useful like "pill"

3. "word_profile()" can then be used to extract the most relevant and prevalent terms (terms which are not in the vocabulary would be ignored)

4. Sentences consisting with these terms would be kept for further analysis, the rest would be discarded. 

#### b. Sentiment analysis of sentences (why should we use it? + initial exploration)
scheme of plan:

![Demo](https://user-images.githubusercontent.com/24357654/43794724-2e0407c2-9a4d-11e8-84d0-243406e62daa.png)

#### Initial exploration conclusions:
##### simple sentiment analysis using a'TextBlob' package showed poor results: 
1. accuracy was low e.g.: "Honey might be another option, as it’s anti-bacterial, but since we’re dealing with a yeast that’s gone wildly out of control, I would think it might be best to avoid even honey at this point. 
 The score for the sentence was positive in a high range (0.55). 

2. The model seems biased towards specific words without considering its content e.g. "Are anti-fungals effective at eliminating candida in the prostate? 0.6" seems biased towards effective. 

3. sentiment analysis might not be the answer to detect useful treatments as advice although could have lower level of sentiment involved.

##### futher attempts at sentiment analysis must be done with the most fitting models out there:

- of sentences and not paragraphs

- of forums which are a dialogue in nature and not a monologue (e.g. reviews)

- of health related.

- neural network on much larger data sets would have the highest potential in showing good results

## Conclusion
NLP models such as LDA and word2vec could be combined to produce powerful understanding of a dialogue based text such as discussion forums about a specific topic dealing with health problems.

word_profile() could be used to "mine" specific terms and give a sense of how important they are in the corpus (their prevalence,topic distribution and the universe of terms they are related to).
by applying word_profile() to symptoms for instance, we could get a sense of the health condition of the users. if I am a user and I type bladder_pain i can see other symptoms that are related (urinary tract and std problems, but also surprisingly prevalent symptom like brain fog. and by typing 'doctor' we can see how people seek naturopath / alternative medicine amongst other specialists in endocrinology, gastrointestinal, dermatologyst and gynecologist. 
