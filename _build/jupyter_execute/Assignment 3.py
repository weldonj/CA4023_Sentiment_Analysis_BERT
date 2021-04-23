#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import tarfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PL04DataLoader_Part_1:
    
    def __init__(self):
        pass
    
    def get_labelled_dataset(self, fold = 0):
        ''' Compile a fold of the data set
        '''
        dataset = []
        for label in ('pos', 'neg'):
            for document in self.get_documents(
                fold = fold,
                label = label,
            ):
                dataset.append((document, label))
        return dataset
    
    def get_documents(self, fold = 0, label = 'pos'):
        ''' Enumerate the raw contents of all data set files.
            Args:
                data_dir: relative or absolute path to the data set folder
                fold: which fold to load (0 to n_folds-1)
                label: 'pos' or 'neg' to
                    select data with positive or negative sentiment
                    polarity
            Return:
                List of tokenised documents, each a list of sentences
                that in turn are lists of tokens
        '''
        raise NotImplementedError

class PL04DataLoader(PL04DataLoader_Part_1):
    
    def get_xval_splits(self):
        ''' Split data with labels for cross-validation
            returns a list of k pairs (training_data, test_data)
            for k cross-validation
        '''
        # load the folds
        folds = []
        for i in range(10):
            folds.append(self.get_labelled_dataset(
                fold = i
            ))
        # create training-test splits
        retval = []
        for i in range(10):
            test_data = folds[i]
            training_data = []
            for j in range(9):
                ij1 = (i+j+1) % 10
                assert ij1 != i
                training_data = training_data + folds[ij1]
            retval.append((training_data, test_data))
        return retval
    
class PL04DataLoaderFromStream(PL04DataLoader):
        
    def __init__(self, tgz_stream, **kwargs):
        super().__init__(**kwargs)
        self.data = {}
        counter = 0
        with tarfile.open(
            mode = 'r|gz',
            fileobj = tgz_stream
        ) as tar_archive:
            for tar_member in tar_archive:
                if counter == 2000:
                    break
                path_components = tar_member.name.split('/')
                filename = path_components[-1]
                if filename.startswith('cv')                 and filename.endswith('.txt')                 and '_' in filename:
                    label = path_components[-2]
                    fold = int(filename[2])
                    key = (fold, label)
                    if key not in self.data:
                        self.data[key] = []
                    f = tar_archive.extractfile(tar_member)
                    document = [
                        line.decode('utf-8').split()
                        for line in f.readlines()
                    ]
                    self.data[key].append(document)
                    counter += 1
            
    def get_documents(self, fold = 0, label = 'pos'):
        return self.data[(fold, label)]

class PL04DataLoaderFromTGZ(PL04DataLoaderFromStream):
    
    def __init__(self, data_path, **kwargs):
        with open(data_path, 'rb') as tgz_stream:
            super().__init__(tgz_stream, **kwargs)


# In[2]:


dir_entries = os.listdir()
dir_entries.sort()


# In[3]:


data_loader = PL04DataLoaderFromTGZ('data.tar.gz')


# In[4]:


# test "get_documents()"

def get_document_preview(document, max_length = 72):
    s = []
    count = 0
    reached_limit = False
    for sentence in document:
        for token in sentence:
            if count + len(token) + len(s) > max_length:
                reached_limit = True
                break
            s.append(token)
            count += len(token)
        if reached_limit:
            break
    return '|'.join(s)
    
for label in 'pos neg'.split():
    print(f'== {label} ==')
    print('doc sentences start of first sentence')
    for index, document in enumerate(data_loader.get_documents(
        label = label
    )):
        print('%3d %7d   %s' %(
            index, len(document), get_document_preview(document)
        ))
        if index == 4:
            break


# In[5]:


# test "get_xval_splits()"

splits = data_loader.get_xval_splits()

print('tr-size te-size (number of documents)')
for xval_tr_data, xval_te_data in splits:
    print('%7d %7d' %(len(xval_tr_data), len(xval_te_data)))


# In[6]:


class PolarityPredictorInterface:

    def train(self, data_with_labels):
        raise NotImplementedError
        
    def predict(self, data):
        raise NotImplementedError


# In[7]:


class PolarityPredictorWithVocabulary(PolarityPredictorInterface):
    
    def train(self, data_with_labels):
        self.reset_vocab()
        self.add_to_vocab_from_data(data_with_labels)
        self.finalise_vocab()
        tr_features = self.extract_features(
            data_with_labels
        )
        tr_targets = self.get_targets(data_with_labels)
        self.train_model_on_features(tr_features, tr_targets)
        
    def reset_vocab(self):
        self.vocab = set()
        
    def add_to_vocab_from_data(self, data):
        for document, label in data:
            for sentence in document:
                for token in sentence:
                    self.vocab.add(token)

    def finalise_vocab(self):
        self.vocab = list(self.vocab)
        # create reverse map for fast token lookup
        self.token2index = {}
        for index, token in enumerate(self.vocab):
            self.token2index[token] = index
        
    def extract_features(self, data):
        raise NotImplementedError
    
    def get_targets(self, data, label2index = None):
        raise NotImplementedError
        
    def train_model_on_features(self, tr_features, tr_targets):
        raise NotImplementedError


# In[8]:


import numpy

class PolarityPredictorWithBagOfWords_01(PolarityPredictorWithVocabulary):
    
    def __init__(self, clip_counts = True):
        self.clip_counts = clip_counts
        
    def extract_features(self, data):
        # create numpy array of required size
        columns = len(self.vocab)
        rows = len(data)
        features = numpy.zeros((rows, columns), dtype=numpy.int32)        
        # populate feature matrix
        for row, item in enumerate(data):
            document, _ = item
            for sentence in document:
                for token in sentence:
                    try:
                        index = self.token2index[token]
                    except KeyError:
                        # token not in vocab
                        # --> skip this token
                        # --> continue with next token
                        continue
                    if self.clip_counts:
                        features[row, index] = 1
                    else:
                        features[row, index] += 1
        return features


# In[9]:


class PolarityPredictorWithBagOfWords(PolarityPredictorWithBagOfWords_01):
 
    def get_targets(self, data):
        ''' create column vector with target labels
        '''
        # prepare target vector
        targets = numpy.zeros(len(data), dtype=numpy.int8)
        index = 0
        for _, label in data:
            if label == 'pos':
                targets[index] = 1
            index += 1
        return targets

    def train_model_on_features(self, tr_features, tr_targets):
        raise NotImplementedError


# ## Naive Bayes
# 
# These next few cells set up the baseline Naive Bayes model that we were supplied with

# In[134]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

class PolarityPredictorBowNB(PolarityPredictorWithBagOfWords):

    def train_model_on_features(self, tr_features, tr_targets):
        # pass numpy array to sklearn to train NB
        self.model = MultinomialNB()
        self.model.fit(tr_features, tr_targets)
        
    def predict(
        self, data, get_accuracy = False,
        get_confusion_matrix = False
    ):
        features = self.extract_features(data)
        # use numpy to get predictions
        y_pred = self.model.predict(features)
        # restore labels
        labels = []
        for i, is_positive in enumerate(y_pred):

            if is_positive:
                labels.append('pos')
            else:
                labels.append('neg')
        if get_accuracy or get_confusion_matrix:
            retval = []
            retval.append(labels)
            y_true = self.get_targets(data)
            if get_accuracy:
                retval.append(
                    metrics.accuracy_score(y_true, y_pred)
                )
            if get_confusion_matrix:
                retval.append(
                    metrics.confusion_matrix(y_true, y_pred)
                )
            return retval, y_true, y_pred
        else:
            return labels, y_pred


# In[135]:


# first functionality test

model = PolarityPredictorBowNB()
model.train(splits[0][0]) 


# In[136]:


predictions, y_true, pred = model.predict(splits[0][1], get_accuracy = True)


# In[137]:


corrects = y_true==pred


# In[138]:


incorrects_nb = []
for i, answer in enumerate(corrects):
    if answer == False:
        incorrects_nb.append(i)


# In[139]:


incorrects_nb


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class PolarityPredictorBowLR(PolarityPredictorWithBagOfWords):

    def train_model_on_features(self, tr_features, tr_targets):
        # pass numpy array to sklearn to train Logistic Regression
        # iterations set to 1000 as default of 100 didn't guarantee convergence with our data
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(tr_features, tr_targets)
        
    def predict(
        self, data, get_accuracy = False,
        get_confusion_matrix = False
    ):
        features = self.extract_features(data)
        # use numpy to get predictions
        y_pred = self.model.predict(features)
        # restore labels
        labels = []
        for is_positive in y_pred:
            if is_positive:
                labels.append('pos')
            else:
                labels.append('neg')
        if get_accuracy or get_confusion_matrix:
            retval = []
            retval.append(labels)
            y_true = self.get_targets(data)
            if get_accuracy:
                retval.append(
                    metrics.accuracy_score(y_true, y_pred)
                )
            if get_confusion_matrix:
                retval.append(
                    metrics.confusion_matrix(y_true, y_pred)
                )
            return retval
        else:
            return labels


# In[140]:


model = PolarityPredictorBowLR()


# In[87]:


bert_256_files = []

for i in range(1,11):
    bert_256_files.append(f'256_BERT/{i}_pred.txt')


bert_512_files = []

for i in range(1,11):
    bert_512_files.append(f'512_BERT/{i}_pred_512.txt')
    

bert_512_12_files = []
    
for i in range(1,11):
    bert_512_12_files.append(f'512_12_BERT/{i}_pred_512_12.txt')    

    
bert_256_neg_files = ['256_BERT_NEG/1_pred_negation.txt']

    
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

df1_neg = pd.DataFrame(columns=c_names)

dataframes_256 = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10]

dataframes_512 = [df1_512,df2_512,df3_512,df4_512,df5_512,df6_512,df7_512,df8_512,df9_512,df10_512]

dataframes_512_12 = [df1_512_12,df2_512_12,df3_512_12,df4_512_12,df5_512_12,df6_512_12,df7_512_12,df8_512_12,df9_512_12,df10_512_12]

dataframes_256_neg = [df1_neg]


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


# In[88]:


dataframes_256 = create_dfs(bert_256_files, dataframes_256)

dataframes_512 = create_dfs(bert_512_files, dataframes_512)

dataframes_512_12 = create_dfs(bert_512_12_files, dataframes_512_12)

dataframes_256_neg = create_dfs(bert_256_neg_files, dataframes_256_neg)


# In[89]:


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


# In[90]:


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


# In[131]:


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


# In[132]:


#error_256 = print_averages_get_errors(dataframes_256, True)
#error_512 = print_averages_get_errors(dataframes_512, True)
error_512_12 = print_averages_get_errors(dataframes_512_12, True)


# In[133]:


error_512[0]


# In[123]:


error_256[0]


# In[125]:


for i in range(len(error_256[0])):
    if error_256[0][i] in error_512[0]:
        print(error_256[0][i])


# In[128]:


for i in range(len(error_512_12[0])):
    if error_512_12[0][i] not in error_256[0]:
        print(error_512_12[0][i])


# In[130]:


print_averages(dataframes_256)


# In[104]:


print_averages(dataframes_512_12)


# In[105]:


print_averages(dataframes_512)


# In[106]:


incorrects_bert = scores[4]


# In[109]:


len(incorrects_bert), len(incorrects_nb)


# In[111]:


for i in range(len(incorrects_nb)):
    if incorrects_nb[i] in incorrects_bert:
        print(incorrects_nb[i])


# In[119]:


df1.iloc[44]['text']


# In[123]:


splits[0][1][44]


# In[126]:


df1.iloc[50]


# In[127]:


df1.iloc[50]['text']


# In[125]:


splits[0][1][50]


# In[128]:


df1.iloc[82]


# In[129]:


df1.iloc[82]['text']


# In[ ]:




