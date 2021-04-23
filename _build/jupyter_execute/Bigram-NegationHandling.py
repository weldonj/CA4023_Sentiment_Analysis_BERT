#!/usr/bin/env python
# coding: utf-8

# ## Bigrams with Negation Handling
# 
# The idea of this section was to use the same setups for the bigram section, but use the dataset created for the negation handling section. 
# 
# The beginning essentially does the set up for this, creating and loading this negation handled dataset and building the model creation structure for bigrams instead of unigrams.
# 
# With this in mind, it's once again best to skip down to the "Naive Bayes" subheading

# In[23]:


import os
import string
pos_path = "./data/txt_sentoken/pos/"
pos_path_out = "./data/txt_sentoken_negation/pos_negation/"
neg_path = "./data/txt_sentoken/neg/"
neg_path_out = "./data/txt_sentoken_negation/neg_negation/"


# In[24]:


def handle_negation(in_path, out_path):
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


# In[25]:


import os
import time
import tarfile
import time

class PL04DataLoader_Part_1:
    
    def __init__(self):
        pass
    
    def get_labelled_dataset(self, fold = 0):
        ''' Compile a fold of the data set
        '''
        dataset = []
        for label in ('pos_negation', 'neg_negation'):
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
    
import tarfile
import time

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
    


class PL04DataLoaderFromFolder(PL04DataLoader):
    
    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        super().__init__(**kwargs)
        
    def get_documents(self, fold = 0, label = 'pos_negation'):
        # read folder contents
        path = os.path.join(self.data_dir, label)
        dir_entries = os.listdir(path)
        # must process entries in numeric order to
        # replicate order of original experiments
        dir_entries.sort()
        # check each entry and add to data if matching
        # selection criteria
        for filename in dir_entries:
            if filename.startswith('cv')             and filename.endswith('.txt'):
                if fold == int(filename[2]):
                    # correct fold
                    f = open(os.path.join(path, filename), 'rt')
                    # "yield" tells Python to return an iterator
                    # object that produces the yields of this
                    # function as elements without creating a
                    # full list of all elements
                    yield [line.split() for line in f.readlines()]
                    f.close()


# In[26]:


dir_entries = os.listdir()
dir_entries.sort()


# In[27]:


data_loader = PL04DataLoaderFromFolder("./data/txt_sentoken_negation/")


# In[28]:


# test "get_documents()"

def get_document_preview(document, max_length = 72):
    s = []
    count = 0
    reached_limit = False
    for sentence in document:
        i = 0
        while (i < len(sentence) - 1):
            token = sentence[i] + ' ' + sentence[i+1]
            if count + len(token) + len(s) > max_length:
                reached_limit = True
                break

            s.append(token)
            count += len(token)
            i+=1
        if reached_limit:
            break
    return '|'.join(s)
    
for label in 'pos_negation neg_negation'.split():
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


# In[29]:


# test "get_xval_splits()"

splits = data_loader.get_xval_splits()

print('tr-size te-size (number of documents)')
for xval_tr_data, xval_te_data in splits:
    print('%7d %7d' %(len(xval_tr_data), len(xval_te_data)))


# In[30]:


class PolarityPredictorInterface:

    def train(self, data_with_labels):
        raise NotImplementedError
        
    def predict(self, data):
        raise NotImplementedError


# In[31]:


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
                i = 0
                while (i < len(sentence) - 1):
                    token = sentence[i] + ' ' + sentence[i+1]
                    self.vocab.add(token)
                    i+=1

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


# In[32]:


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

                i = 0
                while (i < len(sentence)-1):
                    token = sentence[i] + ' ' + sentence[i+1]
                    i+=1

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


# In[33]:


class PolarityPredictorWithBagOfWords(PolarityPredictorWithBagOfWords_01):
 
    def get_targets(self, data):
        ''' create column vector with target labels
        '''
        # prepare target vector
        targets = numpy.zeros(len(data), dtype=numpy.int8)
        index = 0
        for _, label in data:
            if label == 'pos_negation':
                targets[index] = 1
            index += 1
        return targets

    def train_model_on_features(self, tr_features, tr_targets):
        raise NotImplementedError


# ## Naive Bayes

# In[34]:


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
                labels.append('pos_negation')
            else:
                labels.append('neg_negation')
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


# In[35]:


# first functionality test

model = PolarityPredictorBowNB()
model.train(splits[0][0]) 


# In[36]:


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


# In[37]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[38]:


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


# ## Logistic Regression

# In[39]:


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


# In[40]:


model = PolarityPredictorBowLR()
model.train(splits[0][0]) 


# In[41]:


print_first_predictions(model, splits[0][1])


# In[42]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[43]:


print(evaluate_model(model, splits, verbose = True))


# In[ ]:





# ## Decision Tree

# In[44]:


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


# In[45]:


model = PolarityPredictorBowDT()
model.train(splits[0][0])


# In[46]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[47]:


print(evaluate_model(model, splits, verbose = True))


# ## Support Vector Machine

# In[48]:


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


# In[49]:


model = PolarityPredictorBowSVM()
model.train(splits[0][0])


# In[50]:


print_first_predictions(model, splits[0][1])


# In[51]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[52]:


print(evaluate_model(model, splits, verbose = True))


# This combination of negation handling and bigram usage seemed to be ultimately mediocre at best.
# 
# The only algorithm that beat the baseline was the Naive Bayes version and only marginally. Considering the large time cost of running this, particularly for SVM and Decision Tree, it may be best to avoid it.

# In[ ]:




