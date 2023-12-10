import spacy
nlp = spacy.load("en_core_web_sm")
text = (
   "India is my country. "
   "Maharashtra is my state."
)

doc = nlp(text)
print("---------------------Tokenization-----------------------")
for token in doc:
    print(token, token.idx)

doc = nlp(text)
print("\n------------------Stop Words Removal-------------------")
print([token for token in doc if not token.is_stop])

text = "running cats jumping dogs"
doc = nlp(text)
print("\n--------------------Lemmatization-----------------------")
for token in doc:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")

doc = nlp(text)
print("\n------------------Part of Speech Tagging-----------------")
for token in doc:
    print(
        f"""
TOKEN: {str(token)}
=====
TAG: {str(token.tag_):10}
POS: {token.pos_}
EXPLANATION: {spacy.explain(token.tag_)}"""
    )
