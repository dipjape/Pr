import spacy
nlp = spacy.load("en_core_web_sm")
text = "Saurabh is learning Code"
doc=nlp(text)
for token in doc:
  print(
      f"""
TOKEN: {token.text}
=====
{token.tag_= }
{token.head.text= }
{token.dep_= }"""
      )