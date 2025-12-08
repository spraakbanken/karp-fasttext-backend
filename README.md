# Karp Fasttext API

## Production

To be able to minimize the amount of memory this application uses, one can set:

```
SINGLE_MODEL=true
```

And then one model will be loaded at once at most.

Use `NEWER_VERISON=https://spraakbanken4.it.gu.se/karp/fasttext/"` to refer users to a another
instance when they try to use several models in the same request.

Use `MODEL_DIR="<directory>"` to set the directory where models are. The default is `models`.


## Development

Install using `make install`

Add/modify dependencies in `requirements.txt`, use `make update-deps` to install and freeze versions.

Format with `ruff`, using `make fmt-lint` or by invoking `ruff` manually.

Start server with `make serve`.
