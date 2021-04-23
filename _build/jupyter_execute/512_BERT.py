#!/usr/bin/env python
# coding: utf-8

# ## BERT with 512 Max Sequence Length

# In[1]:


import os
import pandas as pd
import numpy as np


# I adjusted the provided model to use a max sequence length of 512 instead of the default 256. This added some significant time to the training process. I saved the predictions from the 10 CV folds so I could repeat the performance evaluation from the previous notebook. These files were each roughly twice as large as the ones from the 256 version (220KB to 430KB) which was to be expected, having roughly twice as much text in each review

# In[4]:


bert_512_files = []

for i in range(1,11):
    bert_512_files.append(f'512_BERT/{i}_pred_512.txt')
    
c_names = ['gold','pred','correct','text']
    
df1_512 = pd.DataFrame(columns=c_names)
df2_512 = pd.DataFrame(columns=c_names)
df3_512 = pd.DataFrame(columns=c_names)
df4_512 = pd.DataFrame(columns=c_names)
df5_512 = pd.DataFrame(columns=c_names)
df6_512 = pd.DataFrame(columns=c_names)
df7_512 = pd.DataFrame(columns=c_names)
df8_512 = pd.DataFrame(columns=c_names)
df9_512 = pd.DataFrame(columns=c_names)
df10_512 = pd.DataFrame(columns=c_names)

dataframes_512 = [df1_512,df2_512,df3_512,df4_512,df5_512,df6_512,df7_512,df8_512,df9_512,df10_512]


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


dataframes_512 = create_dfs(bert_512_files, dataframes_512)


# In[8]:


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


# In[9]:


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


# In[10]:


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

# In[11]:


print_averages_get_errors(dataframes_512, False)


# We can see that this jump from 256 tokens to 512 tokens has given a nice immediate improvement to the average accuracy. We have gone from 89.3% to 91.15%. A not insignificant increase. The next thing I wanted to do was to redo the 512 max sequence length, but double the epoch count to 12. This is what is detailed in the next notebook

# In[ ]:




