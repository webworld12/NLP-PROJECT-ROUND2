
from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk import FreqDist
from matplotlib import pylab as plt
from nltk.tag import pos_tag
from nltk.corpus import brown
from wordcloud import WordCloud, STOPWORDS
from nltk.wsd import lesk
from nltk.sem.relextract import extract_rels, rtuple
import re
import string
import nltk
import en_core_web_sm
import spacy
import pandas as pd
nltk.download('punkt')
nlp = en_core_web_sm.load()
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('maxent_ne_chunker')
nltk.download('words')

#Function for creating a set of (contracted_form,expanded_form) list
def contractions_set():
	contractions_dict = {
		'can\'t': 'cannot',
	'could\'ve': 'could have',
	'couldn\'t': 'could not',
	'didn\'t': 'did not',
	'doesn\'t': 'does not',
	'don\'t': 'do not',
	'hadn\'t': 'had not',
	'hasn\'t': 'has not',
	'haven\'t': 'have not',
	'how\'d': 'how did',
	'how\'ll': 'how will',
	'I\'m': 'I am',
	'I\'ve': 'I have',
	'isn\'t': 'is not',
	'let\'s': 'let us',
	'ma\'am': 'madam',
	'mayn\'t': 'may not',
	'might\'ve': 'might have',
	'mightn\'t': 'might not',
	'must\'ve': 'must have',
	'mustn\'t': 'must not',
	'needn\'t': 'need not',
	'o\'clock': 'of the clock',
	'oughtn\'t': 'ought not',
	'shan\'t': 'shall not',
	'so\'ve': 'so have',
	'they\'re': 'they are',
	'they\'ve': 'they have',
	'to\'ve': 'to have',
	'wasn\'t': 'was not',
	'we\'ll': 'we will',
	'we\'re': 'we are',
	'we\'ve': 'we have',
	'weren\'t': 'were not',
	'what\'re': 'what are',
	'what\'ve': 'what have',
	'when\'ve': 'when have',
	'where\'d': 'where did',
	'where\'ve': 'where have',
	'who\'ve': 'who have',
	'why\'ve': 'why have',
	'will\'ve': 'will have',
	'won\'t': 'will not',
	'would\'ve': 'would have',
	'wouldn\'t': 'would not',
	'y\'all': 'you all',
	'you\'re': 'you are',
	'you\'ve': 'you have'
	}
	return contractions_dict


#Preprossing Function Expands the contractions present in the text
def expand_contractions(text):
	contractions_dict = contractions_set()
	contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
	def replace(match):
	    return contractions_dict[match.group(0)]
	return contractions_re.sub(replace, text)


#Preprocessing Function removes punctuations from the text
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 


#Preprocessing Function convertes all the uppercases to lowercases
def text_lowercase(text): 
    return text.lower() 


#Preprocessing Function removes whitespaces from the text
def remove_whitespaces(text): 
    return  " ".join(text.split()) 


#Preprocessing Function Removes the "Chapter (chapter number)" from the text
def remove_chap(text): 
    result = re.sub(r'CHAPTER \d+', '', text) 
    return result 	


#Preprocessing Function Preprocesses the Book1
def preprocessing_book1(text):
	
	#Removing the title and author name from the text
	text = text.replace( "[Sense and Sensibility by Jane Austen 1811]" , ' ' )
	
	#Removing the end form the text
	text = text.replace("THE END" , ' ' )
	
	#Calling all the prepocessing functions
	#for the book1 text
	text = remove_chap(text)
	text = expand_contractions(text)
	text = remove_punctuation(text)
	text = remove_whitespaces(text)
	text = text_lowercase(text)
	
	#returning the preprocessed text
	return text


#Preprocessing Function Preprocesses the Book2
def preprocessing_book2(text):

	#Removing the title and author name from the text
	text = text.replace( "[Persuasion by Jane Austen 1818]" , ' ' )
	
	#Removing the end form the text
	text = text.replace("Finis" , ' ' )
	
	#Calling all the prepocessing functions
	#for the book2 text
	text = remove_chap(text)
	text = expand_contractions(text)
	text = remove_punctuation(text)
	text = remove_whitespaces(text)
	text = text_lowercase(text)
	
	#returning the preprocessed text
	return text


#Creating a frequency distribution of a given list of tokens
def freq_dist(token):

	fd = FreqDist(token)
	
	#displaying the samples and outcomes
	print(fd)
	
	#displaying the 20 most common words
	print(fd.most_common(20))
	print('\n\n\n\n')
	
	#ploting a line graph of 20 most common words
	fd.plot(20)


#Creating a wordcloud firstly with stopwords then without stopwords
def word_cloud(text):

	stopwords = set(STOPWORDS)
	#clearling the stopwords set so 
	#the word can be made with stopwords
	stopwords.clear()
	
	#creating wordcloud with stopwords
	wordcloud = WordCloud(width = 800, 
							height = 800, 
							background_color ='black', 
							stopwords = stopwords, 
							min_font_size = 10).generate(text) 
	
	#displaying the wordcloud made                  
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(wordcloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0) 
	 
	plt.show() 
	
	#creating the stopwrods set so the 
	#wordcloud is made without them
	stopwords = set(STOPWORDS)
	
	#Creating wordcloud without stopwords
	wordcloud = WordCloud(width = 800, 
							height = 800, 
							background_color ='white', 
							stopwords = stopwords, 
							min_font_size = 10).generate(text) 
	
	#displaying wordcloud                  
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(wordcloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0) 
	 
	plt.show()


#Function for plotting a bar graph
def plot(X,Y,xlabel,ylabel,title):
	plt.bar(X, Y, tick_label = X, width = 0.8, color = ['red', 'green'])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()


#Function to show the relationship between the length of words and their frequency
def plotRelationShipWordLength(fd):

	#fd is the frequency distribution of each of word
	
	#word length is a list that will 
	#store the length of a word and the
	#frequency of the words of that length
	word_lengths = {}
	
	
	for i in fd.keys():
		if len(i) not in word_lengths.keys():
			word_lengths[len(i)] = fd[i]	#adding a new word length and its frequency
		else:
			word_lengths[len(i)] += fd[i]	#adding the frequency of already existing word length
	
	#X will contain all lengths of the word
	#Y will contain the corresponding frequency		
	X = []
	Y = []
	
	for i in word_lengths.keys():
		X.append(i)
		
	X.sort()
	
	for i in X:
		Y.append(word_lengths[i])
		
	#Plotting a bar graph for recorded data
	xlabel = 'word length'
	ylabel = 'frequency'
	title = 'Relationship between word length and frequency'
	plot(X,Y,xlabel,ylabel,title)


#Function for PoS_Tagging
def PoS_Tagging(token):
	
	#PoS tagging done using pos_tag function
	tagged_tuple = pos_tag(token)
	
	#tagged_tuple contains list of (w,t) tuples 
	#where w is the word and 
	#t is the tag for the word
	
	
	#collecting all the tags used 
	tags = [ t for (w,t) in tagged_tuple ]
	
	
	#Analysing frequency distribution of the tags
	freq_dist(tags)


#Function for creating list of sentences for Book 1
def getSentencesbook1(text):

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	#Removing the title and author name from the text
	text = text.replace( "[Sense and Sensibility by Jane Austen 1811]" , ' ' )

	#Removing the end form the text
	text = text.replace("THE END" , ' ' )

	text = remove_chap(text)
	text = expand_contractions(text)
	text = remove_whitespaces(text)
	text = text_lowercase(text)
  
	text = '\n-----\n'.join(tokenizer.tokenize(text))
	sentences = text.split('-----')
 
	pure_sentences = []
	for sentence in sentences:
    # remove punctuation after splitting the sentences
		sentence = remove_punctuation(sentence)
		pure_sentences.append(sentence.replace('\n', ' ').replace('\r',''))
	
	return pure_sentences


#Function for creating list of sentences for Book 2
def getSentencesbook2(text):

	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	#Removing the title and author name from the text
	text = text.replace( "[Persuasion by Jane Austen 1818]" , ' ' )
	
	#Removing the end form the text
	text = text.replace("Finis" , ' ' )

	text = remove_chap(text)
	text = expand_contractions(text)
	text = remove_whitespaces(text)
	text = text_lowercase(text)
  
	text = '\n-----\n'.join(tokenizer.tokenize(text))
	sentences = text.split('-----')
 
	pure_sentences = []
	for sentence in sentences:
    # remove punctuation after splitting the sentences
		sentence = remove_punctuation(sentence)
		pure_sentences.append(sentence.replace('\n', ' ').replace('\r',''))
	
	return pure_sentences


#Function to distribute the words according to the categories of nouns and verbs
def findCategories(tokens,tags,nouns,verbs):
    for word in tokens:
        if not lesk(tokens,word):
            continue
        if lesk(tokens, word).pos() == 'n':
            category = lesk(tokens, word).lexname()
            if category not in nouns.keys():
                nouns[category] = 1
            else:
                nouns[category] += 1
        elif lesk(tokens, word).pos() == 'v':
            category = lesk(tokens, word).lexname()
            if category not in verbs.keys():
                verbs[category] = 1
            else:
                verbs[category] += 1


#Function to find nouns and verbs
def findVerbsAndNouns(sentences):
    nouns = {}
    verbs = {}

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        findCategories(tokens,tags,nouns,verbs)

    return [nouns,verbs]


#Function to plot relationship between categories and their frequency
def plotRelationCategory(nv):
    X = []
    Y = []
    for a in nv.keys():
        X.append(a.split('.')[1][:4])
        Y.append(nv[a])

    
    xlabel = 'Categories'
    ylabel = 'Frequency'
    title = 'Relationship between noun categories and their frequency'
    
    plot(X,Y,xlabel,ylabel,title)


#Function to find nouns and verbs of Books and plotting the relationship
def noun_verb(sentences):

    nouns_verbs = []
    nouns_verbs = findVerbsAndNouns(sentences)

    nouns = nouns_verbs[0]
    verbs = nouns_verbs[1]

    plt.rcParams["figure.figsize"] = [15, 9]

    plotRelationCategory(nouns)

    plotRelationCategory(verbs)


#Functions for entity recognition
def namedEntityRecognition(sentences_book):
    entities = {}
    nlp = en_core_web_sm.load()
    for sentence in sentences_book:
        doc = nlp(sentence)
        for X in doc.ents:
            if X.label_ not in entities.keys():
                entities[X.label_] = []
            entities[X.label_].append(X.text.lower())

    return entities

def Entity_recognition(sentences_book):
    entities = namedEntityRecognition(sentences_book)
    X = []
    Y = []
    for i in entities.keys():
        X.append(i[:4])
        Y.append(len(entities[i]))

    xlabel = 'entities'
    ylabel = 'frequency'
    title = 'Relationship between entities and their frequency'
    plot(X,Y,xlabel,ylabel,title)

def Entities_marked_in_book1():
  doc = nlp(open('book1.txt', encoding="utf8").read())                                
  table = []
  for ent in doc.ents:
      table.append([ent.text,ent.label_,spacy.explain(ent.label_)])
  spacy.displacy.render(doc, style='ent',jupyter=True)

def Entities_marked_in_book2():
  doc = nlp(open('book2.txt', encoding="utf8").read())                                
  table = []
  for ent in doc.ents:
      table.append([ent.text,ent.label_,spacy.explain(ent.label_)])
  spacy.displacy.render(doc, style='ent',jupyter=True)


#Confusion matrix and accuracy
def confusion_matrix_book1(y_pred):
  from sklearn import metrics
  from sklearn.metrics import accuracy_score
  y_act = ['dashwood','old','nephew','years','norland','neice','daughters']
  print(metrics.confusion_matrix(y_act, y_pred, labels= ['old','nephew','neice','daughters','dashwood','norland','three']))
  print(metrics.classification_report(y_act, y_pred, labels= ['old','nephew','neice','daughters','dashwood','norland','three']))
  print(accuracy_score(y_act, y_pred))

def confusion_matrix_book2(y_pred):
  from sklearn import metrics
  from sklearn.metrics import accuracy_score
  y_act = ['Russell','walter','elizabeth','mary','charles musgrove','anne','wayshe','artificial','father','sister','elegance','two']
  print(metrics.confusion_matrix(y_act, y_pred, labels=['second','walter','elizabeth','mary','charles musgrove','anne','wayshe','one','sixteen','two']))
  print(metrics.classification_report(y_act, y_pred, labels=['second','walter','elizabeth','mary','charles musgrove','anne','wayshe','one','sixteen','two']))
  print(accuracy_score(y_act, y_pred))

#Used for findiing Person-Organisation and Person-GPE relationships
def relationBetweenEntities(sentences):
    
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tagged_sentences = [nltk.tag.pos_tag(sentence) for sentence in tokenized_sentences]
    
    OF = re.compile(r'.*\bof\b.*')
    IN = re.compile(r'.*\bin\b(?!\b.+ing)')
    
    print('PERSON-ORGANISATION Relationships:')

    for i, sent in enumerate(tagged_sentences):
        sent = nltk.chunk.ne_chunk(sent) # ne_chunk method expects onetagged sentence
        rels = extract_rels('PER', 'ORG', sent, corpus='ace', pattern=IN, window=10)
        for rel in rels:
            print(rtuple(rel))
    
    print()    
    print('PERSON-GPE Relationships:')

    for i, sent in enumerate(tagged_sentences):
        sent = nltk.chunk.ne_chunk(sent) # ne_chunk method expects one tagged sentence
        rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=OF, window=10)
        for rel in rels:
            print(rtuple(rel))




#Main function starts from here

T1 = open("book1.txt","r")
T1 = T1.read()

T2 = open("book2.txt","r")
T2 = T2.read()

T3 = open("book3.txt","r")
T3 = T3.read()

import numpy as np

corpus = [T1,T2,T3]

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=0.1,stop_words=all_stopwords)
trsfm = vectorizer.fit_transform(corpus)
pnds.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['B1','B2','B3'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(trsfm[0],trsfm)

sentences_book1 = getSentencesbook1(T1)

sentences_book2 = getSentencesbook2(T2)

#Nouns and Verbs
noun_verb(sentences_book1)

noun_verb(sentences_book2)

#Entity Recognition for Book1
Entity_recognition(sentences_book1)

Entities_marked_in_book1()

#Accuracy measure and Confusion Matrix for Book1
T3 = open("sample_text1.txt","r")
T3 = T3.read()
sentences_text3 = getSentencesbook1(T3)
Entities = namedEntityRecognition(sentences_text3)
x = list(Entities.values())

y = []
for i in x:
  for j in i:
    y.append(j)
y
confusion_matrix_book1(y)

#Entity Recognition for Book2
Entity_recognition(sentences_book2)

Entities_marked_in_book2()

#Accuracy measure and Confusion Metrix for Book2
t4 = open("sample_text2.txt","r")
t4 = t4.read()
sentences_text4 = getSentencesbook1(t4)
Entities = namedEntityRecognition(sentences_text4)
x = list(Entities.values())
Entities
y = []
for i in x:
  for j in i:
    y.append(j)
y
confusion_matrix_book2(y)

relationBetweenEntities(sentences_book1)
relationBetweenEntities(sentences_book2)
