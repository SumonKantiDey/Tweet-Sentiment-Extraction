# Tweet-Sentiment-Extraction [link](https://www.kaggle.com/c/tweet-sentiment-extraction/overview).

It is hard to tell whether the sentiment behind a specific text. Suppose I have a text also I know the sentiment label of this text. Now the target of the competition is to **extract support phrases for the sentiment labels**.

```
Text : That`s very funny. Cute kids Sentiment: positive 
Extracted phrase : funny
```
# The metric in this competition is the word-level Jaccard score
```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```
 
# Summary of my approach
```
- Learning Rate scheduling
- Used  Bert-Base-Uncased pre-trained models
- Holdout validation is around 0.59
