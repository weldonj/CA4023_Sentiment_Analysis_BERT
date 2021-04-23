#!/usr/bin/env python
# coding: utf-8

# The first thing I needed to do was replicate the baseline Naive Bayes which was provided to us as part of the assignment and evaluate its performance so I would have something to compare my further experiments to. The next 10 cells are not much different to what was supplied and are simply setting up the data loading etc.
# 
# If you skip to the "Naive Bayes" sub heading, this would be a good place to start.

# In[2]:


import os
import time
import tarfile

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


# In[3]:


dir_entries = os.listdir()
dir_entries.sort()


# In[4]:


data_loader = PL04DataLoaderFromTGZ('data.tar.gz')


# In[5]:


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


# In[6]:


# test "get_xval_splits()"

splits = data_loader.get_xval_splits()

print('tr-size te-size (number of documents)')
for xval_tr_data, xval_te_data in splits:
    print('%7d %7d' %(len(xval_tr_data), len(xval_te_data)))


# In[7]:


class PolarityPredictorInterface:

    def train(self, data_with_labels):
        raise NotImplementedError
        
    def predict(self, data):
        raise NotImplementedError


# In[8]:


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


# In[9]:


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


# In[10]:


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

# In[11]:


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


# In[12]:


# first functionality test

model = PolarityPredictorBowNB()
model.train(splits[0][0]) 


# In[13]:


def print_first_predictions(model, te_data, n = 12):
    predictions = model.predict(te_data)
    for i in range(n):
        document, label = te_data[i]
        prediction = predictions[i]
        print('%4d %s %s %s' %(
            i, label, prediction,
            get_document_preview(document),
        ))
    
print_first_predictions(model, splits[0][1])


# In[17]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# At this point I decided that the "accuracy" mettric that is used in the provided notebook should be improved upon as it can be a misleading value. Precision, Recall and F1 score are better for binary classification tasks as they allow us to account for imbalances in our datasets.
# 
# The cell below contains the function that is used to evaluate model performance and I have edited it from the original so it now reports these additional metrics and its final output is F1 score rather than accuracy

# In[15]:


def evaluate_model(model, splits, verbose = False):
    accuracies = []
    f1s = []
    fold = 0
    for tr_data, te_data in splits:
        if verbose:
            print('Evaluating fold %d of %d' %(fold+1, len(splits)))
            fold += 1
        model.train(tr_data)
        _, accuracy, confusion_matrix = model.predict(te_data, get_accuracy = True, get_confusion_matrix = True)
        
        tp, fp, fn, tn = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        f1 = (2*prec*rec)/(prec+rec)
        
        accuracies.append(accuracy)
        f1s.append(f1)
        if verbose:
            print('Accuracy -->', accuracy)
            print('Precision -->', prec)
            print('Recall -->', rec)
            print('F1 -->', f1)
            print()
    n = float(len(accuracies))
    avg = sum(f1s) / n
    mse = sum([(x-avg)**2 for x in accuracies]) / n
    return (avg, mse**0.5, min(f1s),
            max(f1s))

# this takes about 3 minutes
print(evaluate_model(model, splits, verbose = True))


# The average F1 score after 10 fold cross validation turned out to be **0.834**
# 
# The first step to try improve on this baseline was to use the same unchanged dataset with Logistic Regression instead of Naive Bayes

# ## Logistic Regression
# 
# I imported ScikitLearn's Logistic Regression package for this step. First attempts to use this resulted in the model not converging for some of the 10 folds. 
# 
# This was due to the default iteration number being 100. Once I changed this to 1000 all 10 cross validation folds converged and I was able to get a true average F1 score for this model.

# In[17]:


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


# In[18]:


model = PolarityPredictorBowLR()

print(evaluate_model(model, splits, verbose = True))


# The average F1 score for Baseline Logistic Regression turned out to be **0.867** which was a nice increase on our Baseline Naive Bayes. The runtime was quite similar to the Naive Bayes too so there didn't seem to be any downsides.
# 
# The next implementation that I wanted to try out was using ScikitLearn's Decision Tree algorithm.

# ## Decision Tree

# In[33]:


from sklearn.tree import DecisionTreeClassifier

class PolarityPredictorBowDT(PolarityPredictorWithBagOfWords):

    def train_model_on_features(self, tr_features, tr_targets):
        # pass numpy array to sklearn to train Logistic Regression
        # iterations set to 1000 as default of 100 didn't guarantee convergence with our data
        self.model = DecisionTreeClassifier()
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


# In[34]:


model = PolarityPredictorBowDT()

print(evaluate_model(model, splits, verbose = True))


# The average F1 score for this was **0.610** which was quite a drop from the previous two.
# 
# It also took noticeably longer time to complete the 10 folds so it seemed clear that this was definitely an inferior option.
# 
# Next up was Support Vector Machine implementation.

# ## Support Vector Machine

# In[35]:


from sklearn import svm

class PolarityPredictorBowSVM(PolarityPredictorWithBagOfWords):

    def train_model_on_features(self, tr_features, tr_targets):
        # pass numpy array to sklearn to train Logistic Regression
        # iterations set to 1000 as default of 100 didn't guarantee convergence with our data
        self.model = svm.SVC()
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


# In[36]:


model = PolarityPredictorBowSVM()

print(evaluate_model(model, splits, verbose = True))


# This was more like it. The average F1 score for Baseline SVM was **0.862**. It should be pointed out, however, that the SVM implementation took far longer than the previous ones.
# 
# The overall takeaway at this point was that Decision Trees were not a good choice but that Naive Bayes, Logistic Regression and Support Vector Machine were all viable options with Logistic Regression leading the way.
# 
# The next section details my attempt to add Negation Handling into the mix.

# In[ ]:




