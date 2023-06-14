# Malek Abid reverse dictionary implementation with Sentence Transformers
# https://wn.readthedocs.io/en/latest/index.html
# https://github.com/goodmami/wn#available-wordnets

from txtai.embeddings import Embeddings
import sys
import torch
import json
import wn

__FILES__ = [
    "data/desc.json",
    "data/defi.json",
    "data/unseen.json",
    "data/dev.json",
    "data/train.json",
]

# https://wn.readthedocs.io/en/latest/api/wn.constants.html#wn.constants.PARTS_OF_SPEECH
PARTS_OF_SPEECH = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "r": "adverb",
    "s": "adjective satellite",
    "t": "phrase",
    "c": "conjunction",
    "p": "adposition",
    "x": "other",
    "u": "unknown",
}
wn.download("oewn:2022")
en = wn.Wordnet("oewn:2022")


def load_additional_data(fnames, max_defs=5):
    seen = {}
    moredata = []
    count = 0
    for fname in fnames:
        with open(fname) as f:
            data = json.load(f)
            for item in data:
                word = item["word"]
                if word not in seen:
                    seen[word] = 0
                elif seen[word] <= max_defs:  # 5 definitions max per word
                    seen[word] += 1
                else:
                    continue
                definitions = item["definitions"]
                count += 1
                moredata.append((word, f"{definitions}\nLemmas: {word}\n", None))
    return moredata


# https://wn.readthedocs.io/en/latest/api/wn.html#wn.Synset
def main(name):
    documents = [
        (
            synset.id,
            f"""{synset.definition()}
                Part of speech: {PARTS_OF_SPEECH[synset.pos]}
                Examples: {', '.join(synset.examples())}
                Lemmas: {', '.join(synset.lemmas())} """,
            None,
        )
        for synset in en.synsets()
    ]
    hilldata = load_additional_data(__FILES__)
    documents.extend(hilldata)
    embeddings = Embeddings(
        {
            "path": "sentence-transformers/all-roberta-large-v1",
            "content": True,
            "objects": True,
        }
    )
    embeddings.index(documents)
    embeddings.save("reverse-dictionary-onelook")


if __name__ == "__main__":
    main(sys.argv[1:])
