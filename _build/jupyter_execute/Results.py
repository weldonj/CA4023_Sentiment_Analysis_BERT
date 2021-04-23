#!/usr/bin/env python
# coding: utf-8

# In[71]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

def show_comparison(objects,performance,title):
    objects = objects
    y_pos = np.arange(len(objects))
    performance = performance
    title = title

    objects = [x for _,x in sorted(zip(performance,objects),reverse=True)]
    performance = [x for x,_ in sorted(zip(performance,objects),reverse=True)]

    plt.figure(figsize=(6, 3))
    plt.bar(y_pos, performance, align='center', alpha=0.5, width=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.ylim([0, 1])
    plt.xticks(rotation=45)

    plt.show()


# # *Baseline Setup*

# #### Naive Bayes
# 
# |Average F1|Mean Square Error| Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.83419   | 0.0217            | 0.7942     | 0.8638     |

# #### Logistic Regression
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.867       | 0.0215            | 0.844     | 0.917     |

# #### Decision Tree
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.610       | 0.048            | 0.5     | 0.679     |

# #### Support Vector Machine
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.86       | 0.02459            | 0.825     | 0.9    |

# Now it's time to plot these results as bar charts in sorted order so we can get a quick idea of which algorithm has performed the best

# In[72]:


objects = ('Naive Bayes', 'Logistic Regression', 'Decision Tree', 'SVM')
performance = [0.8341907075166752,0.8670864952717844,0.6103672982111396,0.86]
title = 'Baseline'
show_comparison(objects,performance,title)


# # *Negation Handling Implemented*

# #### Naive Bayes
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.8239       | 0.0195            | 0.788     | 0.857    |

# #### Logistic Regression
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.8639       | 0.0226            | 0.8349     | 0.9126    |

# #### Decision Tree
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.626       | 0.044            | 0.558     | 0.6926    |

# #### Support Vector Machine
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.862       | 0.0243            | 0.828     | 0.904    |

# In[73]:


objects = ('Naive Bayes', 'Logistic Regression', 'Decision Tree', 'SVM')
performance = [0.8239,0.8639,0.62637,0.8624]
title = 'Negation Handling Implemented'
show_comparison(objects,performance,title)


# # *Bigrams instead of Unigrams*

# #### Naive Bayes
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.851       | 0.018            | 0.815     | 0.88    |

# #### Logistic Regression
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.836       | 0.021            | 0.79     | 0.865    |

# #### Decision Tree
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.5965       | 0.0258            | 0.55     | 0.64    |

# #### Support Vector Machine
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.7255       | 0.0248            | 0.69     | 0.77    |

# In[74]:


objects = ('Naive Bayes', 'Logistic Regression', 'Decision Tree', 'SVM')
performance = [0.851,0.836,0.5965,0.7255]
title = 'Bigrams instead of Unigrams'
show_comparison(objects,performance,title)


# # *Trigram*

# #### Naive Bayes
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.839       | 0.01529            | 0.815     | 0.865    |

# #### Logistic Regression
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.7745       | 0.0265            | 0.725     | 0.815    |

# In[75]:


objects = ['Naive Bayes', 'Logistic Regression']
performance = [0.839,0.775]
title = 'Trigram'
show_comparison(objects,performance,title)


# # *Bigrams with Negation Handling*

# #### Naive Bayes
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.8435       | 0.024            | 0.802     | 0.877    |

# #### Logistic Regression
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.8304       | 0.0179            | 0.7920     | 0.8640    |

# #### Decision Tree
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.607       | 0.024            | 0.5297     | 0.6603    |

# #### Support Vector Machine
# 
# | Average F1    | Mean Square Error | Minimum F1 | Maximum F1 |
# | --- | --- | --- | --- |
# | 0.777       | 0.0565            | 0.7509     | 0.8065    |

# In[76]:


objects = ['Naive Bayes', 'Logistic Regression', 'Decision Tree', 'SVM']
performance = [0.8435,0.8304,0.6070,0.77738]
title = 'Bigrams with Negation Handling'
show_comparison(objects,performance,title)


# # *Lexicon*
# 
# 
# The alternate approach of simply using a downloaded pre-built sentiment Lexicon (The Bing Liu Opinion Lexicon) gave the average 
# F1 score of 0.814
# 
# | Average F1    | 
# | --- |
# | 0.814|

# # *All Model Comparison*

# In[79]:


objects = ['Baseline NB', 'Baseline LR', 'Baseline DT', 'Baseline SVM',
          'Negation NB', 'Negation LR', 'Negation DT', 'Negation SVM',
          'Bigram NB', 'Bigram LR', 'Bigram DT', 'Bigram SVM',
          'Trigram NB', 'Trigram LR',
          'Bigram+Negation NB', 'Bigram+Negation LR', 'Bigram+Negation DT', 'Bigram+Negation SVM', 'Lexicon']
y_pos = np.arange(len(objects))
performance = [0.83419,0.86704,0.61036,0.86,
               0.8239,0.8639,0.62637,0.8624,
               0.851,0.836,0.5965,0.7255,
               0.839,0.775,
               0.8435,0.8304,0.6070,0.77738, 0.814]
objects = [x for _,x in sorted(zip(performance,objects),reverse=True)]
performance = [x for x,_ in sorted(zip(performance,objects),reverse=True)]
plt.figure(figsize=(16, 5))
plt.bar(y_pos, performance, align='center', alpha=0.5, width=0.7)
plt.xticks(y_pos, objects)
plt.ylabel('F1 Score')
plt.title('All Models Compared')
plt.ylim([0, 1])
plt.xticks(rotation=45)
plt.show()


# | Ranking    | Model | F1 |
# | --- | --- | --- |
# 1:| Baseline LR |0.8670
# 2:| Negation LR |0.8639
# 3:| Negation SVM |0.8624
# 4:| Baseline SVM |0.86
# 5:| Bigram NB |0.851
# 6:| Bigram+Negation NB |0.8435
# 7:| Trigram NB |0.839
# 8:| Bigram LR |0.836
# **_9:_**| **_Baseline NB_** |**_0.83419_**
# 10:| Bigram+Negation LR |0.8304
# 11:| Negation NB |0.8239
# 12:| Lexicon |0.814
# 13:| Bigram+Negation SVM |0.77738
# 14:| Trigram LR |0.775
# 15:| Bigram SVM |0.7255
# 16:| Negation DT |0.62637
# 17:| Baseline DT |0.61036
# 18:| Bigram+Negation DT |0.607
# 19:| Bigram DT |0.5965
# 

# We can see that Logistic Regression turns out to be the best algorithm for our data set and use case. Interestingly the Baseline version of Logistic Regression with no extra steps seems to outperform any other. I'm not sure if this is due to a mistake in my implementation of Negation Handling and/or the switch to Bigrams. What was pleasing however, was that of the 18 aditional set ups that I implemented, 8 outperformed the Baseline Naive Bayes model that we were given.
# 
# It's clear that for the top 12 or so, there isn't actually a huge difference in performance and all are perfectly viable options. Even the Lexicon set up, which doesn't involve any real training on our end, performs well enough for our use case. The performance really only drops off signifcantly when we use Decision Trees, the four setups that use them take all four of the bottom slots and there's a notable gap up to Bigram SVM in 15th place. 
# 
# In terms of algorithmic performance, SVM was by far the slowest with "Negation SVM" and "Bigram+Negation SVM" both taking almost 12 hours to train. With that in mind, even though two SVM models feature in the top 4 I would rule them out.
# 
# Interestingly, while Bigram did improve the Naive Bayes set up vs. baseline, Trigram took away from performance, and was also incredibly slow. And so I would also rule it out as a viable option despite decent performance. 
# 
# Ultimately I see no real reason to move past the top 2 models in this performance chart, Baseline Logistic Regression and Logistic Regression with Negation Handling.
# 
# One exception to this would be if we didn't have any labelled data to train on in the first place. In this case the only viable option we would have from the options I've experimented with is the Lexicon set up. This doesn't require any labelled data to train on as it is simply referencing a prebuilt sentiment Lexicon.
# 
# In the next section I will detail how to use my implementation of Logistic Regression with the hidden data set.

# In[ ]:




