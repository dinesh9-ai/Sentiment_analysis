from utils import build_freqs,process_tweet
from tkinter import *
import nltk
import numpy as np
from nltk.corpus import twitter_samples
m=Tk()
m.geometry('300x300')
m.title('classify')
t=np.array([[7e-08, 0.0005239, -0.00055517]])
apt=twitter_samples.strings('positive_tweets.json')
ant=twitter_samples.strings('negative_tweets.json')
at=apt+ant
y=np.append(np.ones((len(apt))),np.zeros((len(ant))))
freq=build_freqs(at,y)
def e_t(tw,freq):
    w_l=process_tweet(tw)
    x=np.zeros((1,3))
    x[0,0]=1
    for w in w_l:
        if(w,1.0) in freq:
            x[0,1]+=freq[w,1]
        if (w,0.0) in freq:
            x[0,2]+=freq[w,0]
    assert(x.shape==(1,3))
    return x

def sigmoid(x):
    a=1/(1+np.exp(-x))
    return a
def p_t(tw,t,freq):
    x=e_t(tw,freq)
    pr=sigmoid(np.dot(x,t.T))
    if pr>=0.5:
        return 'positive'
    else:
        return 'negative'

c1=Canvas(m,width=300,height=300)
c1.pack()
e1=Entry(m)
c1.create_window(150,150,window=e1)
def ot(freq=freq,t=t):
    f1=e1.get()
    l1=Label(m,text=p_t(f1,t,freq))
    c1.create_window(150,210,window=l1)
b1=Button(text='get sentiment',command=ot)
c1.create_window(150,180,window=b1)
m.mainloop()
