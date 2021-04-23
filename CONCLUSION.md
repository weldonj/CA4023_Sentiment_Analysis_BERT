# Conclusion

The takeaway message from the work that I did is that BERT is an excellent classifier for sentiment analysis. The results speak for themselves. If the review is short enough where BERT does not have a lot of text pruned at the start, it consistently outperforms the other models. Because most reviews contain a similar theme throughout, this pruning is often also not a big deal. However, ff the reviewer frontloads a positive review with the majority of the positive language BERT can fail to see the review as positive, and likewise if it frontloads a negative review with the majority of the negative language it can fail to see it as negative. 

Naive Bayes and Logistic Regression will not fall foul of this issues as they both examine every token in the review. What they fail to do, however, is pick up any kind of subtle context. While BERT also showed examples of this, it was able to handle it much more often than the others can. This is to be expected due to the way it has been trained.

If a reviewer uses a lot of slang terms or just writes in an uncommon way, all three models can struggle. Again this is to be expected. 

The trade off between quality and training time was an interesting one. It's difficult to decide what is the best choice for each use case, it would need to be considered by the user. The fastest version of the BERT model was still touching 90% accuracy, and far quicker to get through the 10 CV folds vs the the slowest version. It did come at the price of about 2 percentage points however. I would conclude that if time is not an issue, this latter version should be deployed but if the user is in any kind of a hurry, there is little harm in deploying the fastest version. The results will still be fairly excellent.

# Learning Outcome

I felt that I learned a lot from this assignment. Like other assignments where we essentially have to dive in and get out hands dirty, it was impossible not to learn new and useful things. It was nice getting to see the advantage that we would expect BERT to have over the previous models actually manifest in reality. I feel that as a result of this assignment, I would be confident in implementing some useful language models in industry.
