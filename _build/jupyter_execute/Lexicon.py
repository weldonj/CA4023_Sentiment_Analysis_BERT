#!/usr/bin/env python
# coding: utf-8

# # Lexicon Approach
# 
# For this approach I abandoned the previous work and structural setup. I used the sentiment lexicon by Minqing Hu and Bing Liu - Mining and Summarizing Customer Reviews. ACM SIGKDD-2004, found at http://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar
# 
# This contains two .txt files, one with 2006 "Positive" words and one with 4683 "Negative" words. 
# 
# My plan was to store all of the positive words in a list, and all of the negative words in another list. I then take each of the txt files from our dataset and count how many positive words show up in it, and how many negative words show up in it. If there are more positive words than negative words I classified this document as positive and if the opposite were true I classified it as negative.
# 
# I used the negation handling method that we saw earlier to preprocess our data set as I feel it should help when comparing the documents to the sentiment lexicon.

# In[13]:


import os
import string

# Open the two files from the sentiment lexicon, 
# iterate through all of the words and add them to their respective lists
with open('positive-words.txt', 'r') as pos_words, open('negative-words.txt', 'r') as neg_words:
    pos_list = []
    neg_list = []
    for line in pos_words.readlines():
        if (line[0] != ";"):
            pos_list.append(line[:-1])
    for line in neg_words.readlines():
        if (line[0] != ";"):
            neg_list.append(line[:-1])


# In[ ]:


# The function from earlier notebooks that I created to handle negation
# Will add NOT_ to any string that follows another string ending in "n't"

def handle_negation('./data/txt_sentoken_negation/pos_negation/', './data/txt_sentoken_negation/neg_negation/'):
    file_list = os.listdir(in_path)
    for file in file_list:
        new_file = file + "_new.txt"
        new_file_sentences = []
        with open(in_path + file, 'r') as f, open(out_path + new_file, 'w+') as f_out:
            for line in f.readlines():
                new_line = ''
                tokens = line.split()
                i = 0
                while i < len(tokens):

                    if tokens[i][-3:] != "n't":
                        new_line = new_line + tokens[i] + ' '
                        i+=1
                    
                    else:
                        new_line = new_line + tokens[i] + ' '
                        try:
                            while tokens[i+1] not in string.punctuation:
                                new_line = new_line + 'NOT_' + tokens[i+1] + ' '
                                i+=1
                        except:
                            print("end of sentence")
                        i+=1
                new_file_sentences.append(new_line + '\n')
                
            f_out.writelines(new_file_sentences)


# In[24]:


# Will parse the original dataset which has now had negation added, and count how many of the words in each document
# occur in the positive and negative sentiment lexicons. If positive count is higher than negative, its classified as positive

in_path = './data/txt_sentoken_negation/pos_negation/'
file_list = os.listdir(in_path)

total = 0
for file in file_list:
    pos_count = 0
    neg_count = 0
    with open(in_path + file, 'r') as f:
        for line in f.readlines():
            for token in line.split():
                if token in pos_list:
                    pos_count += 1
                elif token in neg_list:
                    neg_count += 1
        if pos_count > neg_count:
            total += 1
                    
print(total/len(file_list))


# This gives us an accuracy of 0.645 (645 of 1000 documents that we know are positive are identified as positive)
# This means we have 645 true positives. As we know there can't be any false positives here we have 0. We know that the remaining documents from the positive 1000 that weren't marked positive must be false negatives, 355. We also know that we cannot have any true negatives here so that is 0. That gives us the following:

# In[31]:


# Simple calculations to get Precision, Recall, F1

tp, fp, fn, tn = 645, 0, 355, 0
prec = tp/(tp + fp)
rec = tp/(tp + fn)
f1_1 = (2*prec*rec)/(prec+rec)


# In[32]:


f1_1


# The F1 score for this was **0.784**
# 
# Now to run the same experiment to see how many of the 1000 negative documents get correctly classified as negative
# 

# In[25]:


# Will parse the original dataset which has now had negation added, and count how many of the words in each document
# occur in the positive and negative sentiment lexicons. If negative count is higher than positive, its classified as negative

in_path = './data/txt_sentoken_negation/neg_negation/'
file_list = os.listdir(in_path)

total = 0
for file in file_list:
    pos_count = 0
    neg_count = 0
    with open(in_path + file, 'r') as f:
        for line in f.readlines():
            for token in line.split():
                if token in pos_list:
                    pos_count += 1
                elif token in neg_list:
                    neg_count += 1
        if pos_count < neg_count:
            total += 1
                    
print(total/len(file_list))


# This gave an accuracy of 0.73 (730 of 1000 documents that we know are negative are identified as negative)
# This means we have 730 true positives. As we know there can't be any false positives here we have 0. We know that the remaining documents from the positive 1000 that weren't marked positive must be false negatives, 270. We also know that we cannot have any true negatives here so that is 0. That gives us the following:

# In[33]:


# Simple calculations to get Precision, Recall, F1

tp, fp, fn, tn = 730, 0, 270, 0
prec = tp/(tp + fp)
rec = tp/(tp + fn)
f1_2 = (2*prec*rec)/(prec+rec)


# In[34]:


f1_2


# The F1 score for this was **0.844**

# In[35]:


(f1_2 + f1_1)/ 2


# Averaging the two F1 scores gave me an overall F1 score for this Lexicon implementation of **0.814** which is fairly competitive when placed against the previous set ups

# In[ ]:




