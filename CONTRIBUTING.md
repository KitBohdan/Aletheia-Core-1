# Contributing
1. Fork → feature branch
2. `python -m pip install -U pip && pip install -e . -r requirements-dev.txt`
3. `ruff check . && black --check .`
4. `pytest -n auto --reruns 2 --reruns-delay 2`
5. Open PR with tests & rationale.