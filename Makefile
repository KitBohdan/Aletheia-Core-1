.PHONY: setup test lint api
setup:
	python -m pip install -U pip
	pip install -e . -r requirements-dev.txt
lint:
	ruff check .
	black --check . || true
	black .
api:
	uvicorn vct.api.app:app --reload --port 8000
test:
	pytest -q
