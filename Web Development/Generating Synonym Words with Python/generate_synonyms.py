import json
from nltk.corpus import wordnet as wn
from collections import defaultdict


with open("synonyms_en.json", "r", encoding="utf-8") as f:
    words_to_process = list(json.load(f).keys())

result = {}

for word in words_to_process:
    pos_dict = defaultdict(set)
    
    for syn in wn.synsets(word):
        pos = syn.pos()
        if pos not in ['n', 'v', 'a', 'r']:  # noun, verb, adj, adv
            continue

        for lemma in syn.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                if pos == 'n':
                    pos_dict['noun'].add(name)
                elif pos == 'v':
                    pos_dict['verb'].add(name)
                elif pos == 'a':
                    pos_dict['adj'].add(name)
                elif pos == 'r':
                    pos_dict['adv'].add(name)

    if pos_dict:
        result[word] = {k: sorted(list(v)) for k, v in pos_dict.items()}


with open("structured_synonyms_by_pos.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Selesai! File saved as structured_synonyms_by_pos.json")
