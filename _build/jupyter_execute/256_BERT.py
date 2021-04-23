#!/usr/bin/env python
# coding: utf-8

# ## BERT as given in assignment

# In[4]:


import os
import pandas as pd
import numpy as np


# The starting point was to take the model that we were given in the assignment and see how it performs. We were given the predictions from the first 3 CV folds so I trained the next 7 and saved all 10 .txt files. The next code blocks will read in the 10 files and get the useful performance metrics from them. Accuracy, Precision, Recall, F1. They also store a list of what reviews the model has classifed incorrectly which will come in useful later on.
# 
# For this baseline mode, there are 6 epochs per CV fold and a max sequence length of 256

# In[5]:


bert_256_files = []

for i in range(1,11):
    bert_256_files.append(f'256_BERT/{i}_pred.txt')
    
c_names = ['gold','pred','correct','text']

df1 = pd.DataFrame(columns=c_names)
df2 = pd.DataFrame(columns=c_names)
df3 = pd.DataFrame(columns=c_names)
df4 = pd.DataFrame(columns=c_names)
df5 = pd.DataFrame(columns=c_names)
df6 = pd.DataFrame(columns=c_names)
df7 = pd.DataFrame(columns=c_names)
df8 = pd.DataFrame(columns=c_names)
df9 = pd.DataFrame(columns=c_names)
df10 = pd.DataFrame(columns=c_names)

dataframes_256 = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]


# In[6]:


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


# In[7]:


dataframes_256 = create_dfs(bert_256_files, dataframes_256)


# In[12]:


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


# In[13]:


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


# In[16]:


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

# In[17]:


print_averages_get_errors(dataframes_256, False)


# It's already clear at this point that the baseline BERT model that we were provided with is an excellent classifier. The average accuracy is just under 90% right out of the box. The next step I wanted to take was to increase the max sequence length from 256 to 512, which I did in the next notebook

# In[ ]:




