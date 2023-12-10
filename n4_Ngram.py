from nltk import ngrams
from nltk.util import ngrams

#unigram model

n = 1
sentence = 'Dr. Vaibhav Lokhande is an expert and experienced General Surgeon, Laparoscopic Surgeon, and Proctologist with an experience of 22 years, and specializes in Proctology, Urology, Pilonidal Sinus, and Laparoscopy. '
unigrams = ngrams(sentence.split(), n)
for item in unigrams:
    print(item)




"""from nltk import ngrams
file = open("/home/exam/Awesome-NLP/Num.txt")
for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
        print("For sentence", counter + 1, ", trigrams are: ")
        trigrams = ngrams(sentence.split(" "), 3)
        for grams in trigrams:
            print(grams)
        counter += 1
        print()
"""