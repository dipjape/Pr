import numpy as np
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim import models

text = "The en_core_web_sm model is a small English language model that comes with spaCy. It's designed to be lightweight and fast, making it suitable for applications where computational resources may be limited. However, it may not capture as many linguistic nuances as larger models."

tokens =[]
for line in text.split('.'):
  tokens.append(simple_preprocess(line, deacc = True))

g_dict = corpora.Dictionary(tokens)

print("The dictionary has: " +str(len(g_dict)) + " tokens\n")
print(g_dict.token2id)

#Using Bag-of-Words(BOW)
g_bow =[g_dict.doc2bow(token, allow_update = True) for token in tokens]
print("Bag of Words : ", g_bow)






"""
import numpy as np
from gensim.utils import simple_preprocess
from gensim import corpora
from gensim import models

text = ["Your smile makes me smile,"
    "Your laugh makes me laugh,"
    "Your eyes are enchanting,"
    "You make my thoughts seem daft."
    "Since the day I first laid eyes on you,"
    "My feelings grew and grew."
    "In that first conversation my knees clicked and clacked,"
    "And those butterflies flipped and flapped."
    "And as I spill these simple rhymes,"
    "My mind goes over time and time,"
    "I have a crush, a little teenage crush"
    "I don't know what to do, about this lovely little crush"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])
"""