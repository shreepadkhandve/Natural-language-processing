#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:45:40 2019

@author: vaishali
"""
import sklearn_crfsuite
import pandas as pd
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
import csv
from itertools import zip_longest

user_input=input('enter sentence:')
a=word_tokenize(user_input)
tagged = nltk.pos_tag(a)
#print('tagged',tagged)
# Using map for 0 index 
a = map(lambda x: x[0], tagged) 
# Using map for 1 index 
b = map(lambda x: x[1], tagged) 
# converting to list 
m = list(a) 
n= list(b)
m.append(0)
n.append(0)
#print(m)
#print(n)
sen=['Sentence:'+str(1)]
d = [sen,m,n]
export_data = zip_longest(*d, fillvalue = '')
with open('test.csv', 'w', encoding="ISO-8859-1", newline='') as myfile: 
    wr = csv.writer(myfile)
    wr.writerow(("Sentence #","Word", "POS","Tag"))
    wr.writerows(export_data)
myfile.close()

df = pd.read_csv('test.csv', encoding = "ISO-8859-1")
#df.head()N
df1 = df.fillna(method='ffill')
#df1.head(10)
token_list=list(df1['Word'])
#token_list
test = df1.drop('Tag', axis=1)
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
                                                           s['POS'].values.tolist(), 
                                                           s['Tag'].values.tolist())]
        
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

### is given word is phone number ?  ##
def isPhoneNumber(word):
    count1=0
    for i in word:
          if(i.isdigit()):
                count1=count1+1
    return count1==10

def word2features(sent, i):
    #print(i)
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isPhoneNumber' : isPhoneNumber(word),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

getter = SentenceGetter(df1)
sent = getter.get_next()
sentences = getter.sentences
test= [sent2features(s) for s in sentences]

from sklearn.externals import joblib 
crf= joblib.load('crf_01_July_2019_optimized_on_best_parameter.pkl')
y_predicted=crf.predict(test)
print(token_list)
print(y_predicted)

lst=y_predicted[0]
lst=['B-Contact_no' if x=='Contact_no' else x for x in lst]
lst=['B-LeaveType' if x=='LeaveType' else x for x in lst]

print(lst)

#print(lst)
#print(token_list)
tokens=[]
lis_var=[]
lis_per=[]
lis_mail=[]
lis_contact=[]
lis_LeaveType=[]
lis_org=[]
lis_Date=[]
lis_tim=[]
lis_gpe=[]
lis_art=[]
lis_eve=[]
lis_nat=[]
lis_Loc=[]
tokens_per=[]
tokens_mail=[]
tokens_contact=[]
tokens_LeaveType=[]
tokens_org=[]
tokens_Date=[]
tokens_tim=[]
tokens_gpe=[]
tokens_art=[]
tokens_eve=[]
tokens_nat=[]
tokens_Loc=[]
for i in range(len(lst)):
    if lst[i]!='O':
        current_entity=lst[i].split('-')[1]
        lis_var.append(current_entity)
        tokens.append(token_list[i])
    #    print(tokens)
# print(lis_var)
# print(tokens)
for i in range(len(lis_var)):
    if(lis_var[i]=='per'):
        tokens_per.append(tokens[i])
        lis_per.append(lis_var[i])
    elif(lis_var[i]=='Email'):
        tokens_mail.append(tokens[i])
    elif(lis_var[i]=='Contact_no'):
        tokens_contact.append(tokens[i])
        lis_contact.append(lis_var[i])
    elif(lis_var[i]=='LeaveType'):
        tokens_LeaveType.append(tokens[i])
        lis_LeaveType.append(lis_var[i])
    elif(lis_var[i]=='org'):
        tokens_org.append(tokens[i])
        lis_org.append(lis_var[i])
    elif(lis_var[i]=='Date'):
        tokens_Date.append(tokens[i])
    elif(lis_var[i]=='tim'):
        tokens_tim.append(tokens[i])
        lis_tim.append(lis_var[i])
    elif(lis_var[i]=='gpe'):
        tokens_gpe.append(tokens[i])
        lis_gpe.append(lis_var[i])
    elif(lis_var[i]=='Email'):
        tokens_Loc.append(tokens[i])
    elif(lis_var[i]=='Loc'):
        tokens_Loc.append(tokens[i])
        lis_Loc.append(lis_var[i])


for i in range(len(tokens_per)):
    print(tokens_per[i],'->',lis_per[i])

tokens_mail1 = [tokens_mail[i] + tokens_mail[i+1] + tokens_mail[i+2] for i in range(0,len(tokens_mail),3)]
for i in range(len(tokens_mail1)):
    lis_mail.append('Email')
for i in range(len(tokens_mail1)):
    print(tokens_mail1[i],'->',lis_mail[i]) 
    
tokens_Date1 = [tokens_Date[i] + ' '+tokens_Date[i+1] +' ' +tokens_Date[i+2] for i in range(0,len(tokens_Date),3)]
for i in range(len(tokens_Date1)):
    lis_Date.append('Date')
for i in range(len(tokens_Date1)):
    print(tokens_Date1[i],'->',lis_Date[i])


for i in range(len(tokens_contact)):
    print(tokens_contact[i],'->',lis_contact[i])
    
for i in range(len(tokens_LeaveType)):
    print(tokens_LeaveType[i],'->',lis_LeaveType[i])
    
for i in range(len(tokens_org)):
    print(tokens_org[i],'->',lis_org[i])
for i in range(len(tokens_tim)):
    print(tokens_tim[i],'->',lis_tim[i])
for i in range(len(tokens_gpe)):
    print(tokens_gpe[i],'->',lis_gpe[i])
for i in range(len(tokens_Loc)):
    print(tokens_Loc[i],'->',lis_Loc[i]) 














