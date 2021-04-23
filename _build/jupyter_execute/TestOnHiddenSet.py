#!/usr/bin/env python
# coding: utf-8

# # Testing Optimal Model on Hidden Set
# 
# To test out what I discovered to be the optimal model through my experimentation on the hidden set mentioned in the assignment spec, simply replace the contents of the two cells underneath "Naive Bayes with SciKit Learn" (in the jupyter notebook that we were provided on loop) with the contents of the two cells below and continue with that notebook to get the evaluation. 
# 
# As I have no idea of the format of the hidden set or the filenames etc. I can't be precise in terms of loading it in but please get in touch with me if there is any issue with implementing this and I will do my best to try and help.

# In[ ]:


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


# In[ ]:


# first functionality test

model = PolarityPredictorBowLR()
model.train(splits[0][0]) 

