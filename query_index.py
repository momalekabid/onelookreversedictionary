from txtai.embeddings import Embeddings
import wn
import readline

# wn.download('oewn:2021')
en = wn.Wordnet('oewn:2022')

embeddings = Embeddings()
embeddings.load('reverse-dictionary')

while True:
        try:
                query = input("> ")
                synset_id = embeddings.search(query, 1)[0]
                #lemmas = en.synset(synset_id).lemmas()
                print(synset_id["text"])
        except KeyboardInterrupt as e:
                print()
                exit()
