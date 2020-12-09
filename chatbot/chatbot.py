import pandas as pd
import os
import json
import nltk
from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize 
from nltk import pos_tag
import numpy as np
import pickle
import string
import random
import timeit
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 

import warnings
warnings.simplefilter('ignore')

# nltk.download('punkt')

convdata = pd.read_csv('COMP3074-CW1-Dataset.csv')

nametalk = pd.read_csv('name_talk.csv')

convdata.drop(['QuestionID','Document'],axis=1)

# Covert dataframes to json
convdata_json = json.loads(convdata.to_json(orient='records'))

nametalk_json = json.loads(nametalk.to_json(orient='records'))
#export as data as JSON
with open('conversation_json.json', 'w') as outfile1:
    json.dump(convdata_json, outfile1)
with open('nametalk_json.json', 'w') as outfile2:
    json.dump(nametalk_json, outfile2)
                           


# Wordnet Lemmatization 
def LemmerTokens(tokens):

    res = []
    lemmatizer = nltk.WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(tokens),tagset='universal'):
        wordnet_pos = get_wordnet_pos(pos)
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN



# Remove punctuation
def RemovePunction(tokens):
    return[t for t in tokens if t not in string.punctuation]


# Create a stopword list from the standard list of stopwords available in nltk
stop_words = set(stopwords.words('english'))
stop_words.add('much')
stop_words.add('many')
stop_words.add('us')


def SelectTask(input_sentence):
    
    tokens = RemovePunction(nltk.word_tokenize(input_sentence.lower()))
    
    filtered_sentence = []
    for w in tokens: 
        filtered_sentence.append(w)  
    
    filtered_sentence =" ".join(filtered_sentence).lower()
    
    word_tokens=LemmerTokens(filtered_sentence)    
    filtered_sentence =" ".join(word_tokens).lower()
            
    test_set = (filtered_sentence,"")

    with open('nametalk_json.json') as sentences_file:

        sentences = []
        reader = json.load(sentences_file)

        for row in reader:
            db_tokens = RemovePunction(nltk.word_tokenize(row['Question'].lower()))

            db_filtered_sentence = [] 
            for dbw in db_tokens: 
                db_filtered_sentence.append(dbw)  

            db_filtered_sentence =" ".join(db_filtered_sentence).lower()
            db_word_tokens = LemmerTokens(db_filtered_sentence)

            db_filtered_sentence =" ".join(db_word_tokens).lower()

            sentences.append(db_filtered_sentence)    

    tfidf_vectorizer = TfidfVectorizer() 
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)
    
    #use the learnt dimension space to run TF-IDF on the query
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)

    # use cosine similarity methods
    cosine = np.delete(cosine_similarity(tfidf_matrix_test, tfidf_matrix_train), 0)
    
    #get the max score
    similarity_max = cosine.max()
    
    if(similarity_max>0.85):
        response_index = 0
        # simply return the index with the highest score
        response_index = np.where(cosine == similarity_max)[0][0]+2

        j = 0 

        with open('nametalk_json.json', "r") as sentences_file:
            reader = json.load(sentences_file)
            for row in reader:
                j += 1 
                if j == response_index: 
                    return row["Document"]
                    break
    else:
        return None



def SmallTalk(instance):
   
    GREETING_RESPONSES_1 = ["Hello!","Hi!","Good day, How may i of help?", "Hello, How can i help?", 
                            "I am glad! You are talking to me.","Good day, How may i of help?", 
                            "Hello, How can i help?", "I am glad! You are talking to me.","Hi! I am good!"]
    GREETING_RESPONSES_2 = ["You are welcome!","My pleasure."]  
    GREETING_RESPONSES_3= time.asctime(time.localtime(time.time()))


    if (instance=='small_talk_1'):
        answer = random.choice(GREETING_RESPONSES_1) 
    elif (instance=='small_talk_2'):
        answer = random.choice(GREETING_RESPONSES_2) 
    elif (instance=='small_talk_3'):
        answer = GREETING_RESPONSES_3
        
    return answer



# Change user name
def ChangeName(sentence):

    new_name = ''
    tokens = RemovePunction(nltk.word_tokenize(sentence))
    new_name = tokens[-1]
    
    return new_name           


def Talk_To_Huey(test_set_sentence):
    json_file_path = "conversation_json.json" 
    tfidf_vectorizer_pickle_path = "tfidf_vectorizer.pkl"
    tfidf_matrix_pickle_path = "tfidf_matrix_train.pkl"
    
    sentences = []
    i=0

    tokens = RemovePunction(nltk.word_tokenize(test_set_sentence.lower()))
    
    filtered_sentence = []
    for w in tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w)  
    
    filtered_sentence =" ".join(filtered_sentence).lower()
    
    word_tokens=LemmerTokens(filtered_sentence)    
    filtered_sentence =" ".join(word_tokens).lower()
            
    test_set = (filtered_sentence,"")
    

    try: 
        # ---------------Use Pre-Train Model------------------#
        f = open(tfidf_vectorizer_pickle_path, 'rb')
        tfidf_vectorizer = pickle.load(f)
        f.close()
        
        f = open(tfidf_matrix_pickle_path, 'rb')
        tfidf_matrix_train = pickle.load(f)
        # ---------------------------------------------------#
    except: 
        # ---------------To Train------------------#
        start = timeit.default_timer()

        with open(json_file_path) as sentences_file:
            reader = json.load(sentences_file)

            # ---------------Tokenisation of training input -----------------------------#    

            for row in reader:
                db_tokens = RemovePunction(nltk.word_tokenize(row['Question'].lower()))

                db_filtered_sentence = [] 
                for dbw in db_tokens: 
                    if dbw not in stop_words: 
                        db_filtered_sentence.append(dbw)  

                db_filtered_sentence =" ".join(db_filtered_sentence).lower()
                db_word_tokens = LemmerTokens(db_filtered_sentence)

                db_filtered_sentence =" ".join(db_word_tokens).lower()

                # #Debugging Checkpoint
                # print('TRAINING INPUT: '+db_filtered_sentence)

                sentences.append(db_filtered_sentence)    
                i +=1
            # ---------------------------------------------------------------------------#

        
        tfidf_vectorizer = TfidfVectorizer() 
        tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)

        #train timing
        stop = timeit.default_timer()
        # print ("Training Time : ")
        # print (stop - start) 

        f = open(tfidf_vectorizer_pickle_path, 'wb')
        pickle.dump(tfidf_vectorizer, f) 
        f.close()

        f = open(tfidf_matrix_pickle_path, 'wb')
        pickle.dump(tfidf_matrix_train, f) 
        f.close 
        # ------------------------------------------#

               
        
    #use the learnt dimension space to run TF-IDF on the query
    tfidf_matrix_test = tfidf_vectorizer.transform(test_set)

    # use cosine similarity methods
    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)
    
    #if not in the topic trained.no similarity 
    flat =  cosine.flatten() #flattened to one dimension.
    flat.sort() # Ascending
    req_tfidf = flat[-2]
    
    if (req_tfidf==0): #Threshold A
        
        not_understood = "Sorry, I am not able to answer this question at the moment."        
        
        return not_understood
        
    else:
        
        cosine = np.delete(cosine, 0)
    

        #get the max score
        similarity_max = cosine.max()
        response_index = 0
        # simply return the index with the highest score
        response_index = np.where(cosine == similarity_max)[0][0]+2

        j = 0 

        with open(json_file_path, "r") as sentences_file:
            reader = json.load(sentences_file)
            for row in reader:
                j += 1 
                if j == response_index: 
                    return row["Answer"]
                    break



flag=True
print("......................................................................................")
print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+ 'My name is Huey.')
print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+ 'I will try my best to answer your question.')
print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+ 'If you want to exit, you can type < bye >.')
print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+ 'By the way, what is your name? My friend.')

name = 'User'

while(flag==True):

    sentence = input('\x1b[0;30;47m' + name +'\x1b[0m'+":")

    if(sentence.lower()!='bye'):
        if(SelectTask(sentence.lower())!=None):
            task=SelectTask(sentence.lower())
            if(task=='change_name'):
                name = ChangeName(sentence)
                print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+ 'Hello! '+name+'.')   
            elif(task=='ask_name'):
                print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+"Your name is "+ name+'.') 
            else:
                print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+SmallTalk(task)) 
        else:
            response_primary = Talk_To_Huey(sentence)
            print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+response_primary)
    else:
        flag=False
        
print('\x1b[1;37;40m' + 'Huey'+'\x1b[0m'+': '+"Bye! Hope that I helped you out!")
print("......................................................................................")





