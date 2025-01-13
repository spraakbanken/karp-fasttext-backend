from fastapi import FastAPI
from gensim.models.fasttext import FastText
from queue import Queue

app = FastAPI()

corpus_lookup = {
    "gp": "gp-2013-2022",
    "dn": "dn-2010-2022",
    "aftonbladet": "afb-2010-2022",
}
types = ["lemma", "token"]


def create_model(corpus, type) -> Queue:
    model_name = f"kubord-fasttext-{corpus_lookup[corpus]}-{type}"
    print(f"loading {model_name}")
    model = FastText.load(f"models/{model_name}/{model_name}.bin")
    q = Queue(maxsize=1)
    q.put(model)
    return q


model_pool = {
    corpus: {type: create_model(corpus, type) for type in types}
    for corpus in corpus_lookup.keys()
}


def get_model(corpus, type):
    pool = model_pool[corpus][type]
    return pool.get(block=True, timeout=5)


def return_model(corpus, type, model):
    model_pool[corpus][type].put(model)


@app.get("/most_similar/{word}")
def read_root(word, corpus=None, type=None):
    model = get_model(corpus, type)
    res = model.wv.most_similar(word, topn=10)
    return_model(corpus, type, model)
    return res
