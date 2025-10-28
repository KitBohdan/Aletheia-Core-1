# VCT RoboDog — Voice Canine Trainer + RoboDog Brain

LLM-first стек для голосового дресурування собак з GPIO-диспенсером винагород і етичними запобіжниками.

## Швидкий старт
```bash
python -m pip install -U pip
pip install -e .
python -m vct.cli --simulate --cmd "сидіти"
```

### Адаптивні голоси TTS

Синтез мовлення тепер конфігурується у `vct/config.yaml`. За замовчуванням використовується сервіс gTTS з україномовним голосом (`com.ua`).

```yaml
tts:
  provider: gtts      # auto | gtts | pyttsx3 | print
  language: uk        # будь-який код, підтримуваний сервісом
  voice: com.ua       # домен верхнього рівня для вибору акценту
  slow: false         # при true читає повільніше
```

Якщо зовнішні залежності недоступні, система автоматично повертається до локального `pyttsx3` або консольного виводу.

## REST API
```bash
uvicorn vct.api.app:app --reload --port 8000
# health: GET /health
# act:    POST /robot/act {"text":"сидіти","confidence":0.9}
```
