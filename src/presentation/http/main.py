"""Точка запуска службы: ``speech-gigaam``."""

import logging

import uvicorn

from src.di.app import build_app
from src.setting.settings import Settings


def main() -> None:
    """Прочитать настройки, собрать приложение и запустить HTTP-сервер.

    Один процесс uvicorn: параллельность дают процессы-обработчики пула.
    Ошибка папки модели останавливает запуск до открытия порта.
    """
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s[%(process)d] %(message)s",
    )
    app = build_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
