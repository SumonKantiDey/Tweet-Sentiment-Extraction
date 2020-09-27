# Tweet-Sentiment-Extraction [link](https://www.kaggle.com/c/tweet-sentiment-extraction/overview).

It is hard to tell whether the sentiment behind a specific text. Suppose I have a text also I know the sentiment label of this text. Now the target of the competition is to **extract support phrases for the sentiment labels**.

```
Text : That`s very funny. Cute kids [Sentiment: positive]
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
- 5-fold model with original sentiment . Val Acc: 0.70.
- CNN layer to enhance feature extraction for span prediction.
- concat last 2 layers of RoBERTa output.
- Postprocessing technique.
```
-- Bert Base Uncased implementation Found [Here](https://github.com/SumonKantiDey/Tweet-Sentiment-Extraction/tree/Bert-Base-Uncased).