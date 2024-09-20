.DEFAULT: serve

.PHONY: serve
serve:
	fastapi dev main.py

.PHONY: run-prod
run-prod:
	fastapi run main.py

.PHONY: install
install:
	pip install -r requirements.lock

.PHONY: update-deps
update-deps:
	pip install -r requirements.txt ; pip freeze -r requirements.txt > requirements.lock

.PHONY: fmt-lint
fmt-lint:
	ruff format .; ruff check --fix .
