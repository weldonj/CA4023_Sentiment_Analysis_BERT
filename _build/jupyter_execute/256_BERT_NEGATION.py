#!/usr/bin/env python
# coding: utf-8

# ## BERT with Negation Handling

# In[2]:


import os
import pandas as pd
import numpy as np


# This was something fairly silly that I wanted to just take a quick look at. In theory, BERT should be able to handle this negation by itself as it can understand the context of negation, having seen many many examples of it in it's training. But with that said, I still felt it would be useful to take a look at how it performed.
# 
# For this I used the documents that I created in the previous assignment with NOT_ added to any tokens that follow an 'n't' and occurr before the next piece of punctuation

# In[3]:


bert_256_neg_files = ['256_BERT_NEG/1_pred_negation.txt']

c_names = ['gold','pred','correct','text']

df1_neg = pd.DataFrame(columns=c_names)

dataframes_256_neg = [df1_neg]


# In[4]:


def create_dfs(files, df_list):
    j = 0
    for dataframe in df_list:

        #dataframe = pd.DataFrame(columns=['index','gold','pred','correct','text'])
        processed_lines = []

        with open(files[j], 'r') as f:
            lines = f.readlines()

            count = 0
            for line in lines[1:]:
                tokens = line.split()
                line_length = len(tokens)
                temp_line = ''

                for i in range(4, (line_length)):
                    temp_line = temp_line + tokens[i] + ' '

                processed_line = [tokens[1],tokens[2],tokens[3], temp_line]
                processed_lines.append(processed_line)
                dataframe.loc[count] = processed_line
                count+=1
        j+=1
    return(df_list)


# In[5]:


dataframes_256_neg = create_dfs(bert_256_neg_files, dataframes_256_neg)


# In[6]:


def get_f1(dataframe):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    corrects = 0
    errors = []
    for i in range(0,len(dataframe)):
        if dataframe.iat[i,2] == 'yes':
            corrects += 1
        else:
            errors.append(i)
        if (dataframe.iat[i,0] == 'pos' and dataframe.iat[i,1] == 'pos'):
            true_pos += 1
        elif (dataframe.iat[i,0] == 'pos' and dataframe.iat[i,1] == 'neg'):
            false_neg += 1
        elif (dataframe.iat[i,0] == 'neg' and dataframe.iat[i,1] == 'neg'):
            true_neg += 1
        elif (dataframe.iat[i,0] == 'neg' and dataframe.iat[i,1] == 'pos'):
            false_pos += 1
    
    accuracy = corrects/len(dataframe)
    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1_score = 2*((precision*recall)/(precision + recall))
    return(accuracy,precision,recall,f1_score,errors)


# In[7]:


def get_averages(df_list):
    accuracies = []
    precs = []
    recs = []
    f1s = []
    errors_list = []
    for dataframe in df_list:    
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        corrects = 0
        errors = []
        for i in range(0,len(dataframe)):
            if dataframe.iat[i,2] == 'yes':
                corrects += 1
            else:
                errors.append(i)
            if (dataframe.iat[i,0] == 'pos' and dataframe.iat[i,1] == 'pos'):
                true_pos += 1
            elif (dataframe.iat[i,0] == 'pos' and dataframe.iat[i,1] == 'neg'):
                false_neg += 1
            elif (dataframe.iat[i,0] == 'neg' and dataframe.iat[i,1] == 'neg'):
                true_neg += 1
            elif (dataframe.iat[i,0] == 'neg' and dataframe.iat[i,1] == 'pos'):
                false_pos += 1

        accuracy = corrects/len(dataframe)
        accuracies.append(accuracy)
        
        precision = true_pos/(true_pos + false_pos)
        precs.append(precision)

        recall = true_pos/(true_pos + false_neg)
        recs.append(recall)
        
        f1_score = 2*((precision*recall)/(precision + recall))
        f1s.append(f1_score)
        
        errors_list.append(errors)
        
    return(sum(accuracies)/len(df_list),sum(precs)/len(df_list),sum(recs)/len(df_list),sum(f1s)/len(df_list), errors_list)


# In[8]:


def print_averages_get_errors(dataframes, errorlist = False):
    acc,prec,rec,f1,errors = get_averages(dataframes)
    if errorlist == True:
        return(errors)
    else:
        for i, dataframe in enumerate(dataframes):
            scores = get_f1(dataframe)
            print(f'Cross validation {i+1}')
            print(f'The accuracy is {scores[0]*100:.2f}%')
            print(f'The precision is {scores[1]*100:.2f}%')
            print(f'The recall is {scores[2]*100:.2f}%')
            print(f'The F1 score is {scores[3]*100:.2f}%')
            print(f'The model got the following rows wrong {scores[4]}\n')

        print(f'The average accuracy is {acc*100:.2f}%')
        print(f'The average precision is {prec*100:.2f}%')
        print(f'The average recall is {rec*100:.2f}%')
        print(f'The average F1 score is {f1*100:.2f}%')


# ## How it performs

# In[10]:


print_averages_get_errors(dataframes_256_neg, False)


# I only performed this experiment on the first CV fold because as I mentioned in the first cell in this notebook, it ultimately does not seem like a very good idea. The results here show that the average accuracy has suffered quite significantly vs the original document set with no negation handling. I would expect this to be the case across the other 9 CV folds. Interestingly, it still outperforms the Baseline NB model from Assignment 2

# In[ ]:




