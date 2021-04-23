#!/usr/bin/env python
# coding: utf-8

# ## BERT with 512 Max Sequence Length and 12 Epochs

# In[1]:


import os
import pandas as pd
import numpy as np


# For this next configuration, I have 512 as the max sequence length and the number of epochs has been doubled from 6 to 12. This was another large jump in training time and it took many hours to complete the 10 CV folds. Once more I saved the ten so I could use the performance evaluation functions that I had written to see how the model performed

# In[10]:


bert_512_12_files = []
    
for i in range(1,11):
    bert_512_12_files.append(f'512_12_BERT/{i}_pred_512_12.txt')    

    
c_names = ['gold','pred','correct','text']
    
df1_512_12 = pd.DataFrame(columns=c_names)
df2_512_12 = pd.DataFrame(columns=c_names)
df3_512_12 = pd.DataFrame(columns=c_names)
df4_512_12 = pd.DataFrame(columns=c_names)
df5_512_12 = pd.DataFrame(columns=c_names)
df6_512_12 = pd.DataFrame(columns=c_names)
df7_512_12 = pd.DataFrame(columns=c_names)
df8_512_12 = pd.DataFrame(columns=c_names)
df9_512_12 = pd.DataFrame(columns=c_names)
df10_512_12 = pd.DataFrame(columns=c_names)

dataframes_512_12 = [df1_512_12,df2_512_12,df3_512_12,df4_512_12,df5_512_12,df6_512_12,df7_512_12,df8_512_12,df9_512_12,df10_512_12]


# In[11]:


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


# In[12]:


dataframes_512_12 = create_dfs(bert_512_12_files, dataframes_512_12)


# In[13]:


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


# In[14]:


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


# In[15]:


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

# In[16]:


print_averages_get_errors(dataframes_512_12, False)


# The average accuracy has jumped once again, from 91.15% to 92.05%. This isnt as big an increase but it's still almost 1 full percentage point, and this configuration ultimately turned out to be the very best performing one across both assignments

# In[ ]:




