# VCT RoboDog — Voice Canine Trainer + RoboDog Brain

[![CI](https://github.com/USER/REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/USER/REPO/actions/workflows/tests.yml)
[![Lint](https://github.com/USER/REPO/actions/workflows/lint.yml/badge.svg)](https://github.com/USER/REPO/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/USER/REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/USER/REPO)

LLM-first стек для голосового дресурування собак з GPIO-диспенсером винагород і етичними запобіжниками.

## Швидкий старт
```bash
python -m pip install -U pip
pip install -e .
python -m vct.cli --simulate --cmd "сидіти"
```

### Адаптивні голоси TTS

Синтез мовлення тепер конфігурується у `vct/config.yaml`. За замовчуванням використовується сервіс gTTS з україномовним голосом
(`com.ua`).

```yaml
tts:
  provider: gtts      # auto | gtts | pyttsx3 | print
  language: uk        # будь-який код, підтримуваний сервісом
  voice: com.ua       # домен верхнього рівня для вибору акценту
  slow: false         # при true читає повільніше
```

Якщо зовнішні залежності недоступні, система автоматично повертається до локального `pyttsx3` або консольного виводу.

## Як запускати тести

### Локально
1. Встановіть дев-залежності:
   ```bash
   python -m pip install -U pip
   pip install -e .[dev]
   ```
2. Запустіть повний набір тестів із покриттям:
   ```bash
   pytest
   ```
   За замовчуванням команда виконує `pytest -ra -q --cov=vct --cov-report=xml`, як визначено у `pyproject.toml`.
3. Для швидкого циклу розробки можна пропускати повільні та БД-тести:
   ```bash
   pytest -m "not slow and not db"
   ```
4. Щоб прогнати лише певну групу, використовуйте мітки:
   ```bash
   pytest -m slow
   pytest -m db
   ```

### CI
- GitHub Actions виконує робочий процес `tests.yml`, який встановлює середовище, запускає `pytest --cov` та публікує артефакт `coverage.xml` для сервісу Codecov.
- Робочий процес `lint.yml` виконує статичний аналіз (ruff/flake8) і має бути зеленим перед злиттям.
- Обидва робочі процеси повинні проходити без попереджень; зламані бейджі вказують на регресії, які треба виправити перед релізом.

### Покриття тестами
- Мінімальний поріг покриття становить **80%** (`fail_under = 80` у секції `[tool.coverage.report]`).
- Якщо покриття падає нижче порогу, локальний прогін `pytest` та перевірка у CI завершаться з помилкою.
- Для локальної перевірки текстового звіту використовуйте:
  ```bash
  pytest --cov-report=term-missing
  ```

### Політика міток `slow` та `db`
- `slow`: призначайте тестам, що працюють >30 секунд або залежать від зовнішніх сервісів/моделей. Вони виконуються щонайменше у nightly-гілках і перед релізом, але за замовчуванням пропускаються локально (`-m "not slow and not db"`).
- `db`: використовуйте для тестів, які потребують реальної бази даних чи складних фікстур. У CI вони запускаються в ізольованому контейнері з попередньо налаштованими секретами.
- Поєднання міток підтримується; описуйте залежності у docstring тесту, щоб полегшити відладку.

## REST API
```bash
uvicorn vct.api.app:app --reload --port 8000
# health: GET /health
# act:    POST /robot/act {"text":"сидіти","confidence":0.9}
```
