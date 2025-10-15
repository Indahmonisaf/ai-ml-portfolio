import json
from collections import defaultdict

file_path = "E:\other\wn-msa-all.tab"  
synset_dict = defaultdict(list)

with open(file_path, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            continue
        
        synset_id, lang, goodness, lemma = parts
        
        if lang != "I":  
            continue
        
        if goodness not in ("Y", "O"): 
            continue
        
        synset_dict[synset_id].append(lemma.lower())

# Bangun sinonim per kata
synonym_dict = defaultdict(set)

for lemmas in synset_dict.values():
    for lemma in lemmas:
        for other in lemmas:
            if lemma != other:
                synonym_dict[lemma].add(other)

# Ubah jadi list dan simpan
result = {word: sorted(list(synonyms)) for word, synonyms in synonym_dict.items()}

with open("synonyms_id.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Selesai! Disimpan ke synonyms_id.json")
