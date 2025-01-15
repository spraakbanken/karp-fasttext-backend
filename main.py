from contextlib import contextmanager
from queue import Queue

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from gensim.models.fasttext import FastText
import uvicorn


newspaper_links = {
    "gp": "https://spraakbanken.gu.se/korp/?mode=kubord#?cqp=%5B%5D&corpus=kubord2-gp-2013,kubord2-gp-2014,kubord2-gp-2015,kubord2-gp-2016,kubord2-gp-2017,kubord2-gp-2019,kubord2-gp-2021,kubord2-gp-2018,kubord2-gp-2020,kubord2-gp-2022&result_tab=2&show_stats&search=word|",
    "dn": "https://spraakbanken.gu.se/korp/?mode=kubord#?cqp=%5B%5D&corpus=kubord2-dn-2010,kubord2-dn-2011,kubord2-dn-2012,kubord2-dn-2013,kubord2-dn-2014,kubord2-dn-2015,kubord2-dn-2016,kubord2-dn-2017,kubord2-dn-2018,kubord2-dn-2019,kubord2-dn-2020,kubord2-dn-2021,kubord2-dn-2022&result_tab=2&show_stats&search=word|",
    "aftonbladet": "https://spraakbanken.gu.se/korp/?mode=kubord#?cqp=%5B%5D&corpus=kubord2-afb-2010,kubord2-afb-2011,kubord2-afb-2012,kubord2-afb-2013,kubord2-afb-2014,kubord2-afb-2016,kubord2-afb-2017,kubord2-afb-2018,kubord2-afb-2019,kubord2-afb-2020,kubord2-afb-2021,kubord2-afb-2015,kubord2-afb-2022&result_tab=2&show_stats&search=word|",
}

newspaper_lookup = {
    "gp": "gp-2013-2022",
    "dn": "dn-2010-2022",
    "aftonbladet": "afb-2010-2022",
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


def get_name(newspaper, type) -> str:
    return f"kubord-fasttext-{newspaper_lookup[newspaper]}-{type}"


def create_model(newspaper, type) -> Queue:
    model_name = get_name(newspaper, type)
    print(f"loading {model_name}")
    model = FastText.load(f"models/{model_name}/{model_name}.bin")
    q = Queue(maxsize=1)
    q.put(model)
    return q


def format_html(newspaper, results, model_name):
    # html is the default format
    tables = []
    for res in results:
        table_rows = [
            f"""<tr>
                    <td>
                        <a href="{newspaper_links[newspaper] + row[0]}" target="_blank">{row[0]}</a>
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


def create_app(model_pool):
    app = FastAPI()

    @contextmanager
    def get_model(newspaper, type):
        pool = model_pool[newspaper][type]
        model = pool.get(block=True, timeout=5)
        try:
            yield model
        finally:
            model_pool[newspaper][type].put(model)

    @app.get("/most_similar/{searches}")
    def read_root(searches, newspaper="all", type="lemma", number=10, format=None):
        searches = searches.split(",")

        if newspaper == "all":
            newspapers = list(newspaper_lookup.keys())
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
                    content.append([model_name, results])
                else:
                    content.append(format_html(newspaper, results, model_name))
        if format == "json":
            return content
        html_content = "".join(content)
        return HTMLResponse(
            content=f"{header}{html_content}</body></html>", status_code=200
        )

    return app


def main():
    # make sure we only have one pool of models, several would consume too much memory
    model_pool = {
        newspaper: {type: create_model(newspaper, type) for type in types}
        for newspaper in newspaper_lookup.keys()
    }
    app = create_app(model_pool)
    print("starting app on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
