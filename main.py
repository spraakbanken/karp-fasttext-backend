from fastapi import FastAPI
from gensim.models.fasttext import FastText

app = FastAPI()

corpus_lookup = {
  "gp": "gp-2013-2022",
  "dn": "dn-2010-2022",
  "aftonbladet": "afb-2010-2022",
}

models = {}

def get_model(corpus, typ):
    model_name = f"kubord-fasttext-{corpus_lookup[corpus]}-{typ}"
    if model_name in models:
        return models[model_name]
    model = FastText.load(f"models/{model_name}/{model_name}.bin")
    models[model_name] = model
    return model
    


@app.get("/most_similar/{word}")
def read_root(word, corpus=None, type=None):
    model = get_model(corpus, type)
    return model.wv.most_similar(word, topn=10)
