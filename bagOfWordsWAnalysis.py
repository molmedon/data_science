import sys
import numpy as np
import nltk as n
#n.download('stopwords')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.pipeline import Pipeline

###load data onto data frame###
df = pd.read_csv('winemag-data-130k-v2.csv')
df = df[np.isfinite(df['price'])]
df = df.reset_index()
###python list from data frame slice###
dscs = df['description'][1:10000]
target = df['points'][1:10000]
#print(target)
###load tokenizer###
tokenizer = n.tokenize.RegexpTokenizer(r'\w+')

######################
#spent some time tokenzing each description but it is just easier to use a pipeline and to produce the bag of words model 
######################
#tokenizer = n.tokenize.RegexpTokenizer(r'\w+')
###tokenize data###
#dscstk = []
#for dsc in dscs:
#  dsc_words = tokenizer.tokenize(dsc)
#  dscstk.append(dsc_words)

###remove stop wordsm, undiscriptive words and make lower case###
#undsc_words = ['wine','fruits','fruit','flavors','drink','aromas','palate']
#undsc_words = ['wine','fruits','fruit','flavors','drink','aromas','palate','finish','acidity','tannins','black','cherry','cherries','oak','ripe','red','rich','spice','notes']
#stop_words = n.corpus.stopwords.words('english')

#filtered_dscs = []
#run_on = []
#for dsc in dscstk:
#  filtered_dsc = []
#  for w in dsc: 
#    wl = w.lower()
#    if wl not in stop_words and wl not in undsc_words: 
#      filtered_dsc.append(wl)
#      run_on.append(wl)
#  filtered_dscs.append(filtered_dsc)
#
#run_on_trg = []
#for trg in target:
#  run_on_trg.append(trg)

#fdist = n.probability.FreqDist(run_on)

#print(fdist)
###create instance of CountVectorizer###
#count_vect = CountVectorizer()
#train_counts = count_vect.fit_trandform()
#fdist.plot(120,cumulative=False)
#plt.show()

###vectorize, remove stop words, make bag of words, and use naive bayes to make a model of points as a fucntion of wine description###
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('clf', BernoulliNB())]) 

text_clf = text_clf.fit(dscs,target)

test_data = df['description'][10001:]
test_target = df['points'][10001:]

predicted = text_clf.predict(test_data)

per = np.mean( np.abs(predicted - test_target) < 2.5)
print(per)
