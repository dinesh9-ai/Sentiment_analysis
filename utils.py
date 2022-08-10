import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def process_tweet(tweet):
    stemmer PorterStemmer()
    stopwords_english=stopwords.words('english')
    t=re.sub(r'\$\w*', '',t)
    t=re.sub(r'^RT[\s]+', '',t)
    t=re.sub(r'https?:\/\/.*[\r\n]*', '',t)
    t=re.sub(r'#', '',t)
    tokenizer=TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
    tweet_tokens=tokenizer.tokenize(t)
    tweets_clean=[]
    for word in tweet_tokens:
        if (word not in stopwords_english and word not in string.punctuation):
            stem_word=stemmer.stem(word)
            tweets_clean.append(stem_word)
    return tweets_clean

def build_freqs(tweets, ys):
    yslist=np.squeeze(ys).tolist()
    freqs={}
    for y,tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            p=(word, y)
            if p in freqs:
                freqs[p] += 1
            else:
                freqs[p] = 1
    return freqs
