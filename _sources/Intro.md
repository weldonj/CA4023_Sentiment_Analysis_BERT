## OVERVIEW

In this assignment we were given a series of options that we could choose from. I have chosen option one, which is to compare the outputs of this BERT model to the baseline model from assignment 2 and also to our best model from assignment 2. We also should take note of what each model has struggled with and what it performs well at.

### Approaches and Structure

**Baseline BERT -**

In this notebook I take a look at the BERT set up that we were given as part of the assignment. I test how it performs across all 10 Cross Validation folds without any changes to the default set up.

**BERT with 512 sequence length -**

Here I increase the maximum sequence length from 256 to 512 and examine what impace this has on the performance, in terms of accuracy but also efficiency.

**BERT with 512 sequence length and 12 Epochs -**

The next adjustment is to increase the training epochs for each CV fold from 6 to 12 and see what impact this has, again on accuracy and also run time and efficiency.

**Baseline BERT with Negation -**

This was a somewhat silly step as there isn't a pressing reason to do it (BERT should handle negation already). I basically take the document set, process it for negation handling, and run it through the baseline BERT.

**BERT vs Assignment 2 Performance -**

In this notebook I lay out all of the models against each other to see which one ultimately performs the best. I also take an in depth look at the actual review text that they struggle on.

**Conclusion and Learning Outcomes -**

Here I wrap things up with a conclusion for the assignment itself and also list some learning outcomes from the whole process.