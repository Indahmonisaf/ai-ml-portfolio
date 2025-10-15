import json
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')

data = {}
for synset in wn.all_synsets():
    for lemma in synset.lemmas():
        word = lemma.name().lower().replace('_', ' ')
        synonyms = [l.name().replace('_', ' ') for l in synset.lemmas()]
        if word not in data:
            data[word] = list(set(synonyms))

with open("synonyms_en.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
