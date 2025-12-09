from contextlib import contextmanager
import os
from queue import Queue
import re
import sys
from typing import Optional
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from gensim.models.fasttext import FastText
from pydantic import BaseModel
import uvicorn


class NewspaperSetting:
    model_name: str
    corpora: str

    def __init__(self, newspaper, year_from, year_to):
        self.model_name = f"{newspaper}-{year_from}-{year_to}"
        self.corpora = ",".join(f"kubord2-{newspaper}-{year}" for year in range(year_from, year_to + 1))

    def link(self, type, val):
        if type == "lemma":
            encoded = urllib.parse.quote(f'[(word = "{val}" %c | lemma contains "{val}")]')
            query = f"search_tab=1&within=word&search=cqp&cqp={encoded}"
        else:
            query = f"&isCaseInsensitive&search=word|{val}"
        return f"https://spraakbanken.gu.se/korp/?mode=kubord#?corpus={self.corpora}&result_tab=2&show_stats&{query}"


def init_newspaper_settings():
    settings = {}
    for filename in Path(os.getenv("MODEL_DIR", "models")).glob("*token*"):
        print(str(filename.name))
        regexp = r"kubord-fasttext-([a-z]+)-([0-9]{4})-([0-9]{4})-token"
        groups = re.match(regexp, str(filename.name)).groups()
        newspaper = groups[0]
        year_from = int(groups[1])
        year_to = int(groups[2])
        settings[newspaper] = NewspaperSetting(newspaper, year_from, year_to)
    return settings


newspaper_settings = init_newspaper_settings()


types = ["lemma", "token"]


header = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>kubord-fasttext results</title>
    </head>
    <body>
"""

api_description = """Exposes the fasttext models under <https://spraakbanken.gu.se/resurser/kubord-fasttext>.
as an API, with support for fetching the most similar words according to their word vector representation.

There are six models available (three newspapers, two types of models)."""


most_similar_description = """Calls fasttext `most_similar` on the selected model once per search query (see `searches`).

For more information, see the
[gensim documentation](https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastTextKeyedVectors.most_similar)"""


class SearchResult(BaseModel):
    search_forms: list[str]
    table: list[tuple[str, float]]


class ModelResult(BaseModel):
    fasttext_model_name: str
    results: list[SearchResult]


def get_name(newspaper, type) -> str:
    return f"kubord-fasttext-{newspaper_settings[newspaper].model_name}-{type}"


def create_model(newspaper, type) -> FastText:
    model_name = get_name(newspaper, type)
    print(f"loading {model_name}")
    model_dir = os.getenv("MODEL_DIR", "models")
    return FastText.load(str(Path(model_dir) / model_name / f"{model_name}.bin"))


def create_model_queue(newspaper, type) -> Queue:
    model = create_model(newspaper, type)
    q = Queue(maxsize=1)
    q.put(model)
    return q


def format_html(
    newspaper: str, results: list[tuple[list[str], list[tuple[str, float]]]], model_name: str, type: str
) -> str:
    tables = []
    for res in results:
        table_rows = [
            f"""<tr>
                    <td>
                        <a href="{newspaper_settings[newspaper].link(type, row[0])}" target="_blank">{row[0]}</a>
                    </td>
                    <td>{row[1]}</td>
                </tr>"""
            for row in res[1]
        ]
        title = ", ".join(res[0])
        table = "<h3>" + title + "</h3><table>" + "".join(table_rows) + "</table>"
        tables.append(table)
    table_str = "".join(tables)
    return f"<h2>{model_name}</h2>{table_str}"


def format_json(_, results: list[tuple[list[str], list[tuple[str, float]]]], model_name: str, __: str) -> ModelResult:
    return ModelResult(
        fasttext_model_name=model_name,
        results=[SearchResult(search_forms=result[0], table=result[1]) for result in results],
    )


def create_app(model_pool, single_model=False, newer_version=None):
    """
    - model_pool holds pre-created model object
    - single_model if true, there is only one model loaded at a time
    """
    root_path = os.environ.get("ROOT_PATH")
    app = FastAPI(
        title="Språkbanken kubord-fasttext API",
        description=api_description,
        root_path=root_path,
        redoc_url="/",
        docs_url=None,
    )

    @contextmanager
    def get_model(newspaper, type):
        """
        Fasttext models are not thread-safe, so only let one request use each model at a time
        """
        pool = model_pool[newspaper][type]
        model = pool.get(block=True, timeout=5)

        if isinstance(model, tuple):
            # running in single model mode
            model_newspaper, model_type, current_model = model
            try:
                new_ref = None
                if model_newspaper != newspaper or model_type != type:
                    # this will hopefully make the memory return faster
                    del current_model
                    new_ref = create_model(newspaper, type)
                else:
                    new_ref = current_model
                yield new_ref
            finally:
                if new_ref is not None:
                    model_pool[newspaper][type].put((newspaper, type, new_ref))
                else:
                    model_pool[newspaper][type].put(("", "", None))
        else:
            try:
                yield model
            finally:
                model_pool[newspaper][type].put(model)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ):
        detail = f"<div>{exc.detail}</div>"
        if newer_version:
            full_without_host = request.url.path
            if request.url.query:
                full_without_host += "?" + request.url.query
            detail += f'<div>Try: <a href="{newer_version}{full_without_host}">{newer_version}{full_without_host}</a> for full functionality.</div>'
        if request.headers.get("accept") == "application/json":
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": detail},
            )
        else:
            return HTMLResponse(status_code=exc.status_code, content=f"{header}{detail}</body></html>")

    @app.get("/most_similar/{searches}", response_model=None | ModelResult, description=most_similar_description)
    def most_similar(
        searches: str = Path(
            description="A comma-separated list of searches. Each search may contain more than one wordform, separated by |",
            examples="hållbarhet,hållbarhet|material",
        ),
        type: str = Query("lemma", description="""Model type, can be either "lemma" or "token"."""),
        number: int = Query(
            10,
            description="The number of results to return.",
        ),
        newspaper: Optional[str] = Query(
            None,
            description="""Comma-separated list of newspapers
                               Can be omitted to search in all newspapers.
                               Available newspapers are: gp, dn or aftonbladet.""",
        ),
        accept: Optional[str] = Header(
            None, description="set to application/json for JSON format, otherwhise HTML is returned"
        ),
    ) -> Response:
        searches = searches.split(",")

        if not newspaper:
            newspapers = list(newspaper_settings.keys())
        else:
            newspapers = newspaper.split(",")
        if single_model and len(newspapers) > 1:
            raise HTTPException(status_code=400, detail="This instance supports only one newspaper per query")

        content = []
        for newspaper in newspapers:
            with get_model(newspaper, type) as model:
                results = []
                for search in searches:
                    # words are separated with "|"
                    words = search.split("|")
                    res = model.wv.most_similar(positive=words, topn=int(number))
                    results.append((words, res))

                model_name = get_name(newspaper, type)
                if accept == "application/json":
                    fun = format_json
                else:
                    fun = format_html
                content.append(fun(newspaper, results, model_name, type))
        if accept == "application/json":
            return JSONResponse(content=jsonable_encoder(content))
        html_content = "".join(content)
        return HTMLResponse(content=f"{header}{html_content}</body></html>", status_code=200)

    return app


def main():
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 8000

    # the maximum number of models to load
    single_model = os.getenv("SINGLE_MODEL")
    # a reference to a newer version
    newer_version = os.getenv("NEWER_VERSION")

    newspapers = newspaper_settings.keys()
    if not single_model:
        # make sure we only have one pool of models, by creating it before creating the app (which can have many threads)
        model_pool = {
            newspaper: {type: create_model_queue(newspaper, type) for type in types} for newspaper in newspapers
        }
    else:
        # if running with SINGLE_MODEL set, we do not pre-load the models, but use a single Queue for all combinations of newspaper & type as a lock
        lock = Queue(maxsize=1)
        lock.put(("", "", None))
        model_pool = {newspaper: {type: lock for type in types} for newspaper in newspapers}

    app = create_app(model_pool, single_model=single_model, newer_version=newer_version)
    print(f"starting app on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
