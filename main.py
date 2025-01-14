from contextlib import contextmanager
import logging
from queue import Queue

from fastapi import FastAPI
from gensim.models.fasttext import FastText

app = FastAPI()

newspaper_lookup = {
    "gp": "gp-2013-2022",
    "dn": "dn-2010-2022",
    "aftonbladet": "afb-2010-2022",
}

logger = logging.Logger("main")

types = ["lemma", "token"]


def create_model(newspaper, type) -> Queue:
    model_name = f"kubord-fasttext-{newspaper_lookup[newspaper]}-{type}"
    logger.info(f"loading {model_name}")
    model = FastText.load(f"models/{model_name}/{model_name}.bin")
    q = Queue(maxsize=1)
    q.put(model)
    return q


model_pool = {
    newspaper: {type: create_model(newspaper, type) for type in types}
    for newspaper in newspaper_lookup.keys()
}


@contextmanager
def get_model(newspaper, type):
    pool = model_pool[newspaper][type]
    model = pool.get(block=True, timeout=5)
    try:
        yield model
    finally:
        model_pool[newspaper][type].put(model)


@app.get("/most_similar/{word}")
def read_root(word, newspaper=None, type=None):
    with get_model(newspaper, type) as model:
        return model.wv.most_similar(word, topn=10)
