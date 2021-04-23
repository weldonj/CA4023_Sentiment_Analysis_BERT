#!/usr/bin/env python
# coding: utf-8

# ## Bigram Implementation
# 
# To use Bigrams instead of Unigrams I had to make some changes to how the vocabulary was built. In the cell below it can be seen that the tokens are now appended as pairs of strings instead of single strings

# In[4]:


# test "get_documents()"

def get_document_preview(document, max_length = 72):
    s = []
    count = 0
    reached_limit = False
    for sentence in document:
        i = 0
        
        # This while loop will ensure that we append pairs of strings as our tokens rather than single strings
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


# Again, this version of the PolarityPredictorWithVocabulary class will now append pairs of strings instead of single strings

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

# In[10]:


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


# In[11]:


# first functionality test

model = PolarityPredictorBowNB()
model.train(splits[0][0]) 


# In[12]:


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


# In[13]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[14]:


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


# This F1 score of **0.845** was an improvement over the Baseline NB implementation

# ## Logistic Regression

# In[15]:


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


# In[16]:


model = PolarityPredictorBowLR()
model.train(splits[0][0]) 


# In[17]:


print_first_predictions(model, splits[0][1])


# In[18]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[19]:


print(evaluate_model(model, splits, verbose = True))


# This F1 score of **0.84** was worse than the Baseline Logistic Regression

# ## Decision Tree

# In[20]:


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


# In[21]:


model = PolarityPredictorBowDT()
model.train(splits[0][0])


# In[22]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[23]:


print(evaluate_model(model, splits, verbose = True))


# ## Support Vector Machine

# In[15]:


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


# In[16]:


model = PolarityPredictorBowSVM()
model.train(splits[0][0])


# In[17]:


labels, accuracy, confusion_matrix = model.predict(
    splits[0][1], get_accuracy = True, get_confusion_matrix = True
)

print(accuracy)
print(confusion_matrix)


# In[18]:


print(evaluate_model(model, splits, verbose = True))


# This F1 Score for SVM of **0.77** was worse than Baseline SVM
# 
# Once again, implementing Bigrams didn't guarantee an improvment in the performance of any individual algorithm but it did improve some, at the cost of a slower run time.
# 
# The next step was to examine what a move to Trigrams might do.

# In[ ]:




