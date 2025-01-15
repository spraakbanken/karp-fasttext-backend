from contextlib import contextmanager
from dataclasses import dataclass
from queue import Queue
import sys
from typing import Optional

from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from gensim.models.fasttext import FastText
from pydantic import BaseModel
import uvicorn


@dataclass
class NewspaperSetting:
    link: str
    model_name: str


newspaper_settings = {
    "gp": NewspaperSetting(
        link="https://spraakbanken.gu.se/korp/?mode=kubord#?cqp=%5B%5D&corpus=kubord2-gp-2013,kubord2-gp-2014,kubord2-gp-2015,kubord2-gp-2016,kubord2-gp-2017,kubord2-gp-2019,kubord2-gp-2021,kubord2-gp-2018,kubord2-gp-2020,kubord2-gp-2022&result_tab=2&show_stats&search=word|",
        model_name="gp-2013-2022",
    ),
    "dn": NewspaperSetting(
        link="https://spraakbanken.gu.se/korp/?mode=kubord#?cqp=%5B%5D&corpus=kubord2-dn-2010,kubord2-dn-2011,kubord2-dn-2012,kubord2-dn-2013,kubord2-dn-2014,kubord2-dn-2015,kubord2-dn-2016,kubord2-dn-2017,kubord2-dn-2018,kubord2-dn-2019,kubord2-dn-2020,kubord2-dn-2021,kubord2-dn-2022&result_tab=2&show_stats&search=word|",
        model_name="dn-2010-2022",
    ),
    "aftonbladet": NewspaperSetting(
        link="https://spraakbanken.gu.se/korp/?mode=kubord#?cqp=%5B%5D&corpus=kubord2-afb-2010,kubord2-afb-2011,kubord2-afb-2012,kubord2-afb-2013,kubord2-afb-2014,kubord2-afb-2016,kubord2-afb-2017,kubord2-afb-2018,kubord2-afb-2019,kubord2-afb-2020,kubord2-afb-2021,kubord2-afb-2015,kubord2-afb-2022&result_tab=2&show_stats&search=word|",
        model_name="afb-2010-2022",
    ),
}


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


class SearchResult(BaseModel):
    search_forms: list[str]
    table: list[tuple[str, float]]


class ModelResult(BaseModel):
    fasttext_model_name: str
    results: list[SearchResult]


def get_name(newspaper, type) -> str:
    return f"kubord-fasttext-{newspaper_settings[newspaper].model_name}-{type}"


def create_model(newspaper, type) -> Queue:
    model_name = get_name(newspaper, type)
    print(f"loading {model_name}")
    model = FastText.load(f"models/{model_name}/{model_name}.bin")
    q = Queue(maxsize=1)
    q.put(model)
    return q


def format_html(newspaper: str, results: list[tuple[list[str], list[tuple[str, float]]]], model_name: str) -> str:
    tables = []
    for res in results:
        table_rows = [
            f"""<tr>
                    <td>
                        <a href="{newspaper_settings[newspaper].link + row[0]}" target="_blank">{row[0]}</a>
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


def format_json(_, results: list[tuple[list[str], list[tuple[str, float]]]], model_name: str) -> ModelResult:
    return ModelResult(
        fasttext_model_name=model_name,
        results=[SearchResult(search_forms=result[0], table=result[1]) for result in results],
    )


def create_app(model_pool):
    app = FastAPI()

    @contextmanager
    def get_model(newspaper, type):
        """
        Fasttext models are not thread-safe, so only let one request ues each model at a time
        """
        pool = model_pool[newspaper][type]
        model = pool.get(block=True, timeout=5)
        try:
            yield model
        finally:
            model_pool[newspaper][type].put(model)

    @app.get("/most_similar/{searches}", response_model=None | ModelResult)
    def read_root(
        searches: str, newspaper: str = "all", type: str = "lemma", number: int = 10, format: Optional[str] = None
    ) -> Response:
        searches = searches.split(",")

        if newspaper == "all":
            newspapers = list(newspaper_settings.keys())
        else:
            newspapers = newspaper.split(",")

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
                if format == "json":
                    fun = format_json
                else:
                    fun = format_html
                content.append(fun(newspaper, results, model_name))
        if format == "json":
            return JSONResponse(content=jsonable_encoder(content))
        html_content = "".join(content)
        return HTMLResponse(content=f"{header}{html_content}</body></html>", status_code=200)

    return app


def main():
    # for testing purposes, call main.py with one newspaper, to make startup faster
    if len(sys.argv) > 1:
        newspapers = [sys.argv[1]]
    else:
        newspapers = newspaper_settings.keys()
    # make sure we only have one pool of models, several would consume too much memory
    model_pool = {newspaper: {type: create_model(newspaper, type) for type in types} for newspaper in newspapers}
    app = create_app(model_pool)
    print("starting app on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
