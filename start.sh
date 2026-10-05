#!/bin/zsh
# Запуск веб-интерфейса БЗПО: ./start.sh
# Поднимает Postgres.app и Ollama, если они не запущены, и открывает страницу в браузере.
set -e
cd "$(dirname "$0")"

PY=./.venv/bin/python
if [[ ! -x $PY ]]; then
  echo "Нет окружения .venv — см. docs/LAPTOP.md, раздел «Установка»"; exit 1
fi

export BZPO_DB__DSN=${BZPO_DB__DSN:-postgresql://localhost/bzpo}
# Модели уже скачаны: не ходить на Hugging Face при каждом запуске.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

PG_BIN=/Applications/Postgres.app/Contents/Versions/latest/bin
if ! $PG_BIN/pg_isready -q 2>/dev/null; then
  echo "Запускаю Postgres.app…"
  open -ga Postgres
fi

if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
  echo "Запускаю Ollama…"
  # Экономия памяти на 16 ГБ: одна модель в памяти, сжатый KV-кэш (docs/LAPTOP.md).
  launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
  launchctl setenv OLLAMA_FLASH_ATTENTION 1
  launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
  open -ga Ollama
fi

for _ in {1..30}; do
  $PG_BIN/pg_isready -q 2>/dev/null && curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break
  sleep 1
done

echo "Открываю http://localhost:8501 (остановить — Ctrl+C)"
# Слежение за файлами выключено: перебирая модули, Streamlit дёргает ленивые
# импорты transformers и сыплет ошибками про torchvision, который не нужен.
# После правки кода страницу перезапускают (Ctrl+C, ./start.sh).
exec $PY -m streamlit run bzpo/ui/app.py \
  --server.port 8501 --browser.gatherUsageStats false --server.headless false \
  --server.fileWatcherType none
