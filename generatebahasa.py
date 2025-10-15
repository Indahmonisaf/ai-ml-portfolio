import json
from collections import defaultdict
import csv


file_path = "wn-msa-all.tab"  

# Mapping POS dari suffix synset
pos_map = {
    'n': 'noun',
    'v': 'verb',
    'a': 'adj',
    'r': 'adv'
}

# Langkah 1: Parsing data Wordnet Bahasa
synset_dict = defaultdict(list)

with open(file_path, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 4:
            continue
        synset_id, lang, quality, lemma = parts

        # Fokus hanya untuk Bahasa Indonesia
        if lang != "M" or quality not in ("Y", "O"):
            continue

        pos_suffix = synset_id.split("-")[-1]
        pos = pos_map.get(pos_suffix)
        if not pos:
            continue

        synset_dict[(synset_id, pos)].append(lemma.lower())

# Langkah 2: Bangun struktur sinonim
csv_rows = []
structured_by_pos = defaultdict(lambda: defaultdict(set))

for (synset_id, pos), lemmas in synset_dict.items():
    for lemma in lemmas:
        for synonym in lemmas:
            if synonym != lemma:
                structured_by_pos[lemma][pos].add(synonym)
                csv_rows.append({
                    "word": lemma,
                    "synonym": synonym,
                    "pos": pos,
                    "language": "ms"
                })

# Langkah 3: Simpan ke JSON
result = {
    word: {pos: sorted(list(synonyms)) for pos, synonyms in pos_map.items()}
    for word, pos_map in structured_by_pos.items()
}

with open("structured_synonyms_id_by_pos_MS.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Selesai! File tersimpan sebagai structured_synonyms_id_by_pos_MS.json")
