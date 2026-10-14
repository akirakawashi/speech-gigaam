# Служба speech-gigaam: процессор, без CUDA. Видеокарта нужна только контейнеру Triton.
FROM python:3.12-slim-bookworm AS build

RUN pip install --no-cache-dir uv==0.11.18
# UV_CONCURRENT_INSTALLS: по умолчанию uv компилирует по процессу на ядро; на сервере их 128,
# и сборка падает на лимите открытых файлов (os error 24). Четырёх хватает.
ENV UV_COMPILE_BYTECODE=1 \
    UV_CONCURRENT_INSTALLS=4 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

FROM python:3.12-slim-bookworm
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /app
COPY --from=build /app /app
# Проверка сборки: тяжёлые библиотеки импортируются, модель поиска речи на месте.
RUN python -c "import av, onnxruntime, sentencepiece, torch, torchaudio, tritonclient.grpc; \
from src.infrastructure.vad import silero_model_path; silero_model_path(); \
assert not torch.cuda.is_available()"
USER 10001:10001
EXPOSE 8000
CMD ["speech-gigaam"]
