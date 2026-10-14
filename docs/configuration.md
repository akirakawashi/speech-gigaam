# Настройки speech-gigaam

Все настройки — переменные окружения с приставкой `GIGAAM_`. Compose берёт их из
`.env` (образец — [`.env.example`](../.env.example)).

## Compose

| Переменная | По умолчанию | Что задаёт |
|---|---|---|
| `GIGAAM_MODEL_DIR` | — | папка модели на сервере; обязательна ([модель](model.md)) |
| `GIGAAM_GPU` | `0` | номер видеокарты для Triton |
| `GIGAAM_PORT` | `8090` | порт службы на `127.0.0.1` |
| `GIGAAM_CPUS` | `8` | ядра контейнера службы; около 2 на обработчик |
| `GIGAAM_MEMORY` | `16g` | память контейнера службы |
| `GIGAAM_TRITON_IMAGE` | `nvcr.io/nvidia/tritonserver:25.06-py3` | образ Triton в локальном Docker |
| `GIGAAM_UID`, `GIGAAM_GID` | `10001` | пользователь контейнеров |
| `GIGAAM_VERSION` | `0.1.0` | тег образа службы |

## Служба

| Переменная | По умолчанию | Что задаёт |
|---|---|---|
| `GIGAAM_WORKERS` | `4` | число процессов-обработчиков, то есть записей одновременно |
| `GIGAAM_THREADS_PER_WORKER` | `2` | потоки математических библиотек в обработчике |
| `GIGAAM_QUEUE_LIMIT` | = `GIGAAM_WORKERS` | сколько запросов ждут сверх занятых обработчиков; дальше — 503 |
| `GIGAAM_REQUEST_TIMEOUT_S` | `600` | предел ожидания обработчика и предел обработки записи |
| `GIGAAM_WORKER_START_TIMEOUT_S` | `180` | предел загрузки моделей одним обработчиком |
| `GIGAAM_RETRY_AFTER_S` | `2` | подсказка `Retry-After` в ответах 503 |
| `GIGAAM_MAX_UPLOAD_MB` | `200` | предел размера файла |
| `GIGAAM_MAX_AUDIO_SECONDS` | `7200` | предел длительности записи |
| `GIGAAM_TRITON_URL` | `triton:8001` | адрес gRPC Triton; в compose задан |
| `GIGAAM_TRITON_MODEL` | `gigaam_encoder` | имя модели энкодера в Triton |
| `GIGAAM_TRITON_TIMEOUT_S` | `60` | предел одного вызова энкодера |
| `GIGAAM_MODEL_DIR` | `/model` | папка модели внутри контейнера; в compose задана |
| `GIGAAM_VERIFY_ENGINE` | `true` | сверять SHA256 движка с `движок.json` при старте (1–2 с) |
| `GIGAAM_LOG_LEVEL` | `INFO` | уровень журнала |
| `GIGAAM_HOST`, `GIGAAM_PORT` | `0.0.0.0`, `8000` | адрес внутри контейнера |

## Поиск речи

| Переменная | По умолчанию |
|---|---|
| `GIGAAM_VAD_THRESHOLD` | `0.5` |
| `GIGAAM_VAD_MIN_SILENCE` | `0.45` |
| `GIGAAM_VAD_MIN_SPEECH` | `0.25` |
| `GIGAAM_VAD_MAX_SPEECH` | `15.0` (не больше 24) |

Это те же значения, что на стенде `temptest`. От них зависят границы реплик и текст:
после изменения результат уже не сравнить со стендом.
